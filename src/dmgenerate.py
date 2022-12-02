import os
import fire
import torch
import numpy as np
import pdb
import dmutil
from src.utils.helpers import to_cuda_variable, to_numpy

# Load a DMelodies model, and then perform one or more of the following
# operations:

# 1) test - test the model by extracting latent codes and the corresponding
# attributes for "test" data segment

# 2) generate - save latent codes and attributes for "test" data segment to
# a file for processing outside the DMelodies framework. <generate> names
# the output directory, and a ".npz" file is created with the same name as
# the original model.

# 3) validate - check the performance of the VAE by encoding and then
# decoding a batch of test samples. <validate> is the size of the batch (eg.
# --validate 2000). <block> defines how many times to repeat each sample.
# Original latent codes are saved to "latent.csv" and attributes to
# "attributes.csv". Decoded attributes are saved to "decoded.csv". Prints
# the proportion of the decoded samples that have only valid attributes, and
# saves these values to "valid.csv". For each sample a MIDI file is also
# generated containing the original score (before encode/decoding), and the
# <block> separate decodings of the corresponding original latent code.

# 4) scores - generate scores for all "test" samples. MIDI scores are named
# by sequence number "score-NNNNNN.mid", and stored in the directory
# specified (eg. --scores midi_dir).

# 5) decode - decode latent codes and compute the corresponding features,
# and repeat this for each of N features. For each feature F in 0..N-1, we
# read the file "input-F.csv" and write the file "output-F.csv". The
# parameters specify input, output files, and N, in a single argument
# separated by commas like this (eg. --decode
# "input-{}.csv,output-{}.csv,9"). File names can contain the {} pattern to
# substitute feature index. If --midi is specified, MIDI scores will be
# saved to the specified directory.

# Parameters
#
# path - path to SeNT model directory
#
# generate, decode, validate, scores - operations as described above
#
# midi - directory to optionally save midi scores (for 'decode' operation)
# 
# block - number of times to repeat each sample (for 'validate' operation)
#
# split - parameters define the partitioning of data for
# train/validate/testing. For all operations this will determine the
# directory from which to load the source DMelodies VAE in the situation
# where there are multiple models for different. For operations that
# enumerate test samples, this determines the particular test samples to use
# (test, validate, generate, scores operations). Only the 'decode' operation
# does not depend on this parameter.
#
# threads - if defined, this limits the number of threads to use for
# processing. Otherwise, all threads will be used. Use this to optimise the
# running of several tasks in parallel on the same machine.
#
# batch_size - the batch size to use when enumerating test samples

def loadModel(path, generate=None, decode=None, midi=None, validate=None, scores=None, block=2, test=None, split=None, threads=None, batch_size=256):
	name = os.path.basename(path)
	print("split", split)
	if threads is not None:
		# optionally limit the number of threads
		# if not specified PyTorch will attempt to use ALL available cores,
		# causing problems when several jobs run in parallel
		torch.set_num_threads(threads)

	print("generate", generate, "decode", decode, "validate", validate)

	vae_trainer = dmutil.openModel(path, split)
	dataset = vae_trainer.dataset

	header = ",".join(dataset.latent_dicts.keys())

	if test is not None:
		print("test")
		print("Data size is", len(dataset.dataset))

		a, b, c = vae_trainer.dataset.data_loaders(batch_size=batch_size, split=split)
		print("a", len(a), len(a)*batch_size)
		print("b", len(b), len(b)*batch_size)
		print("c", len(c), len(c)*batch_size)
		latent_codes, attributes, attr_list = vae_trainer.compute_representations(c, num_batches=len(c))
		print("latent_codes", latent_codes.shape)
		print("attributes", attributes.shape)
		return

	if validate is not None:
		_, _, gen_test = vae_trainer.dataset.data_loaders(batch_size=validate, split=split)
		batch = next(iter(gen_test))
		score, attributes = batch
		s = []
		a = []

		for i in range(len(score)):
			for j in range(block):
				s.append(score[i].numpy())
				a.append(attributes[i].numpy())

		s = np.vstack([s])
		a = np.vstack([a])

		data = [(torch.from_numpy(s), torch.from_numpy(a))]

		print("Encoding...")
		latent_codes, attributes, names, inputs = vae_trainer.compute_representations(data, num_batches=1, return_input=True)
		np.savetxt("attributes.csv", attributes, delimiter=",", header=header, comments="", fmt='%d')
		np.savetxt("latent.csv", latent_codes, delimiter=',')

		print("Decoding...")
		score, score_tensor = vae_trainer.decode_latent_codes(torch.from_numpy(latent_codes))
		labels = [ vae_trainer.compute_attribute_labels(score_tensor[i])[0] for i in range(score_tensor.size(0)) ]
		np.savetxt(f"decoded.csv", labels, delimiter=',', header=header, comments="", fmt='%d')

		print("Validating...")
		nLabels = len(labels[0])
		valid = np.zeros((nLabels))
		for l in labels:
			for i in range(nLabels):
				if l[i] != -1:
					valid[i] += 1
		valid = valid / validate
		print("valid", valid)
		np.savetxt(f"valid.csv", [ [ name ] + list(valid)], delimiter=',', header="name,"+header, comments="", fmt="%s")
		print("Generating midi...")
		st = score_tensor.numpy()
		st = [ i[0] for i in st ]
		np.savetxt(f"scoretensor.csv", st, delimiter=',', fmt='%d')
		for i in range(validate):
			score = torch.cat((torch.from_numpy(inputs[block*i]), score_tensor[block*i:block*(i+1),:].flatten()))
			s = vae_trainer.dataset.tensor_to_m21score(score)
			s.write('midi', fp=f'score-{i:04d}.mid')

	# -- generate
	# Convert test data set to latent codes "z.csv" and attributes "f.csv"

	if generate is not None:
		outPath = generate
		_, _, gen_test = vae_trainer.dataset.data_loaders(batch_size=batch_size, split=split)
		print(f"Extracting {len(gen_test)} batches")
		latent_codes, attributes, attr_list = vae_trainer.compute_representations(gen_test, num_batches=len(gen_test))
		print("latent_codes", latent_codes.shape)
		print("attributes", attributes.shape)
		np.savez(os.path.join(outPath, f"{name}.npz"), latent_codes=latent_codes, attributes=attributes, fNames=list(dataset.latent_dicts.keys()))

	# extract MIDI scores for each test sample
	if scores is not None:
		try:
			os.mkdir(scores)
		except FileExistsError:
			pass
		_, _, gen_test = vae_trainer.dataset.data_loaders(batch_size=batch_size, split=split)
		seq = 0
		print("Generating midi scores...");
		for batch in gen_test:
			score, attributes = batch
			print(f"seq={seq}")
			for i in range(len(score)):
				s = vae_trainer.dataset.tensor_to_m21score(score[i])
				s.write('midi', fp=os.path.join(scores, f'score-{seq:06d}.mid'))
				seq += 1
		print("Done")


	if decode is not None:
		if midi is not None:
			try:
				os.mkdir(midi)
			except FileExistsError:
				pass

		source, dest, count = tuple(decode.split(","))
		for f in range(int(count)):
			inName = source.format(f)
			outName = dest.format(f)
			print(f"Decoding {inName} -> {outName}")

			samples = np.loadtxt(inName, delimiter=',', dtype=np.float32)
			print("samples", samples.shape)
			score, score_tensor = vae_trainer.decode_latent_codes(to_cuda_variable(torch.from_numpy(samples)))
			score_tensor = score_tensor.cpu()
			print("score", score)
			print("score_tensor", score_tensor.shape)
			if midi is not None:
				for i in range(len(samples)):
					s = vae_trainer.dataset.tensor_to_m21score(score_tensor[i])
					s.write('midi', fp=os.path.join(midi, f'{i:05d}-{f}-result.mid'))
				# s = vae_trainer.dataset.tensor_to_m21score(score_tensor[0:100,:].flatten())
				# print("score full...")
				# s.write('midi', fp=os.path.join(midi, f'score-{f}-full.mid'))
	
			labels = [ vae_trainer.compute_attribute_labels(score_tensor[i])[0] for i in range(score_tensor.size(0)) ]
			np.savetxt(outName, labels, delimiter=',', header=header, comments="", fmt='%d')

fire.Fire(loadModel)

