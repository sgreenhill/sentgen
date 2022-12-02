import numpy as np
import fire
from tqdm import tqdm
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import util
import dmutil
from src.utils.helpers import to_cuda_variable

epochs = 400
batch_size = 64
device="cpu"

class ZtoC(nn.Module):
	def __init__(self, width, outWidth, scale=4):
		super(ZtoC, self).__init__()
		self.args = (width, outWidth, scale)

		nodes = []
		
		i = width
		o = int(width * scale)
		while o >= 16:
			print("i", i, "o", o)
			nodes.append(nn.Linear(i, o))
			nodes.append(nn.ReLU())
			i = o
			o = o // 2

		nodes.append(nn.Linear(i, outWidth))
		nodes.append(nn.Sigmoid())

		self.model = nn.Sequential(*nodes)

	def forward(self, z):
		return self.model(z)

	def save(self, fileName):
		torch.save({
			'args' : self.args,
			'state' : self.state_dict()
		}, fileName)

	def load(fileName):
		data = torch.load(fileName)
		net = ZtoC(*data['args'])
		net.load_state_dict(data['state'])
		net.eval()
		return net

# SentNet - neural network for SeNT algorithm
#
# cWidth - Feature width, normally 1. Except when using 1-hot encoding
# where cWidth is a vector with length the number of values of the feature.
#
# width - width of output vector. Normally, the number of dimensions D of the
# latent space to be traversed.
#
# inWidth - this is normally D, the number of latent dimensions. To increase
# the number of parameters inWidth may be increased to say 2*D. Internally the
# net will maintain the larger width until the final stage which always outputs
# width=D 

class SentNet(nn.Module):

	def __init__(self, cWidth, width, inWidth=None, preStages=0, withTarget=False):
		super(SentNet, self).__init__()
		if inWidth is None:
			inWidth = width
		self.args = (cWidth, width, inWidth, preStages, withTarget)
		self.preStages = preStages
		self.withTarget = withTarget
		self.inputC1 = nn.Linear(cWidth, inWidth)
		self.inputC2 = nn.Linear(cWidth, inWidth)
		self.inputZ = nn.Linear(width, inWidth*2)
		catWidth = inWidth * (4 if withTarget else 3)
		self.cat = nn.Linear(catWidth, inWidth*2)
		self.reduce = nn.Linear(inWidth*2, width)

		self.pre1 = [ nn.Linear(inWidth, inWidth) for i in range(preStages) ]
		self.pre2 = [ nn.Linear(inWidth, inWidth) for i in range(preStages) ]

		# nodes must be named in Module to be treated as trainable parameters
		for i in range(self.preStages):
			setattr(self, f"pre1-{i}", self.pre1[i])
			setattr(self, f"pre2-{i}", self.pre2[i])

	def forward(self, z, c1, c2):
		inZ = F.relu(self.inputZ(z))
		inC1 = F.relu(self.inputC1(c1))
		target = inC2 = F.relu(self.inputC2(c2))
		for i in range(self.preStages):
			inC1 = F.relu(self.pre1[i](inC1))
			inC2 = F.relu(self.pre2[i](inC2))
		diff = torch.sub(inC1, inC2)

		c = torch.cat(((diff, inZ, target) if self.withTarget else (diff, inZ)), 1)
		x = F.relu(self.cat(c))
		x = F.tanh(self.reduce(x))

		return x

	def save(self, fileName):
		torch.save({
			'args' : self.args,
			'state' : self.state_dict()
		}, fileName)

	def load(fileName):
		data = torch.load(fileName)
		net = SentNet(*data['args'])
		net.load_state_dict(data['state'])
		net.eval()
		return net

# SentDataset - pytorch wraper for SeNT training data
#
# Data consists of pairs of samples representing an attribute change, where
# each samle has a latent space position (z) and an attribute (c) of the
# generated object. 
#
# The first pair (z1, c1) represents a reference point.
#
# The second pair (z2, c2) represents a point the value of the attribute has
# changed from c1 to c2, while other attributes are relatively unchanged.

class SentDataset(Dataset):
	def __init__(self, z1, c1, c2, z2):
		self.z1 = z1.astype(np.float32)
		self.z2 = z2.astype(np.float32)
		self.c1 = c1.astype(np.float32)
		self.c2 = c2.astype(np.float32)
		print("z1", self.z1.shape, self.z1.dtype)
		print("z2", self.z2.shape, self.z2.dtype)
		print("c1", self.c1.shape, self.c1.dtype)
		print("c2", self.c2.shape, self.c2.dtype)

	def __len__(self):
		return len(self.z1)

	def __getitem__(self, idx):
		return (self.z1[idx], self.c1[idx], self.c2[idx]), self.z2[idx]

def showModel(model, w=50):
	total = 0
	print("-"*w)
	for name, param in model.named_parameters():
		p = np.prod(param.size())
		type = "trainable" if param.requires_grad else ""
		print(f"{type:10} {name:15} {p:7} {param.dtype} {list(param.size())}")
		total += p
	print("-"*w)
	print(f"total {total} parameters")
	print("-"*w)

# return mean squared difference, but only for dimensions in <indices>

def similarForIndex(yPred, yTrue, indices):
	yPred = yPred[:,indices]
	yTrue = yTrue[:,indices]
	return torch.mean((yPred - yTrue)**2)

# Sometimes loss is a float, and sometimes a Tensor. WHY?

def getFloat(value):
	return value if isinstance(value, float) else value.item()

# given a set of latent codes, quickly return the feature labels
# 
# feature - if defined, return just the feature at this index
# otherwise, return the entire feature vector

def decodeFeatures(vaeTrainer, latentCodes, feature=None):
	# this is a modified vaeTrainer.decode_latent_codes() to avoid m21score generation
	# m21score generation is too slow to be used in the neural network training loop
	# this code should be functionally equivalent to:
	#	score, score_tensor = vaeTrainer.decode_latent_codes(z0)
	#	labels = [ vaeTrainer.compute_attribute_labels(score_tensor[i])[0][f] for i in range(score_tensor.size(0)) ]
	#	return vaeTrainer.compute_attribute_labels(tensor_score)

	batch_size = latentCodes.size(0)
	dummy_score_tensor = to_cuda_variable(torch.zeros(batch_size, 16))
	decoder = dmutil.getDecoder(vaeTrainer)
	_, tensor_score = decoder(latentCodes, dummy_score_tensor, False)

	labels = np.array([ vaeTrainer.compute_attribute_labels(tensor_score[i])[0] for i in range(tensor_score.size(0)) ])
	if feature is not None:
		labels = labels[:,feature]
	return labels

# decodeFeatures replaces this implementation

def decodeFeatures0(vaeTrainer,latentCodes):
	score, score_tensor = vaeTrainer.decode_latent_codes(latentCodes)
	labels = [ vaeTrainer.compute_attribute_labels(score_tensor[i])[0] for i in range(score_tensor.size(0)) ]
	return np.array(labels)

# return the name of the SeNT model for feature

def modelFileName(path, feature, subType=""):
	if subType != "":
		subType += "_"
	return os.path.join(path, f"sent_{subType}{feature}.model")

def setThreads(threads):
	if threads is not None:
		print(f"set session for {threads} threads")
		torch.set_num_threads(threads)

def oneHotDecode(value):
	for i in range(len(value)):
		if value[i] > 0:
			return i
	return -1

# train_neural_traversal - train the SeNT model
#
# path - path to the directory containing source data and results
#   model files will be output to this directory
#   the files corr_index and uncorr_index must exist for all features
#   other source files are accessed through util.Reader
#
# lambda_1, lambda_2, lambda_3 - are the weights for the three components of
# the loss function
#
# limit - if defined, process only this many training samples
#
# threads - if defined, use only this many threads
#
# oneHot - if defined, use one-hot encoding for categorical feature
#
# iscale - use iscale*D length tensors for internal nodes, where D is the
# number of latent dimensions of the target space. Set this > 1 to increase the
# number of parameters
#
# batch_size - batch size to use during training
#
# epochs - number of epochs to use during training
#
# lr - learning rate to use during training
#
# features - which features to train. Default (None) is to train all features.
# One model will be generated for each feature
#
# profile - when True, profile the training run, saving results to NeuralTraversal2.prof
# 
# split - specify alternate data split for source VAE model, which must have
# been already generated

def train_neural_traversal(path, lambda_1, lambda_2, lambda_3, limit=None, threads=None, oneHot=False, iscale=1, preStages=0, withTarget=False, batch_size=batch_size, epochs=epochs, lr=0.001, features=None, profile=False, split=None, withZC=True, simulateLossC=False):

	if profile:
		import cProfile, pstats
		profiler = cProfile.Profile()
		profiler.enable()

	setThreads(threads)

	rd = util.openSource(path, oneHot=oneHot)

	# set up denormalisation for Z
	zMin = torch.tensor(rd.config['sRange'][0])
	zMax = torch.tensor(rd.config['sRange'][1])
	zSpan = zMax - zMin
	print('zMin', zMin)
	print('zMax', zMax)
	print('zSpan', zSpan)
	def denormaliseZ(z):
		return 0.5*(z+1)*zSpan+zMin

	# open VAE model
	source = rd.config['source']
	vaeTrainer = dmutil.openModel(source, split=split)

	# which features to train?
	if features is None:
		features = range(rd.nFeatures)

	for f in features:
		print(f"Feature {f}")
		modelSaveFile = modelFileName(path, f)
		# set up normalisation for C
		cMin = rd.config['fRange'][0][f]
		cMax = rd.config['fRange'][1][f]
		cSpan = cMax - cMin

		relevantIndex = list(np.loadtxt(f"{path}/corr_index_{f}.txt").astype(int).flatten())
		irrelevantIndex = list(np.loadtxt(f"{path}/uncorr_index_{f}.txt").astype(int).flatten())

		# read training data, with limited length if requested
		Z_1, C_1, Z_s, C_s = rd.read(f)
		if limit is not None:
			Z_1 = Z_1[:limit]
			Z_s = Z_s[:limit]
			C_1 = C_1[:limit]
			C_s = C_s[:limit]

		print("z", Z_1.shape, "c", C_1.shape)
	
		N = C_1.shape[0]
		width = Z_1.shape[1]
		cWidth = C_1.shape[1]
		iwidth = width * iscale	

		# create SeNT network
		model = SentNet(cWidth, width, inWidth=iwidth, preStages=preStages, withTarget=withTarget)
		showModel(model)

		# load ZC network for loss gradient
		if withZC:
			zc = ZtoC.load(modelFileName(path, f, "ZC"))
			for param in zc.parameters():
				pass #param.requires_grad = False
			showModel(zc)

		# prepare to optimise trainable parameters
		opt=torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr)

		# create a DataLoader for requested batch size
		dataset = SentDataset(Z_1, C_1, C_s, Z_s)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		log = open(os.path.join(path,f"train_{f}.csv"), "w")

		bestLoss = 1e6
		for epoch in tqdm(range(epochs)):
			for (z1, c1, c2), z2 in loader:
				opt.zero_grad()
				pred = model(z1, c1, c2)

				lossr = similarForIndex(pred, z2, relevantIndex) 
				lossi = similarForIndex(pred, z1, irrelevantIndex)

				if withZC and simulateLossC:
					c = zc(pred)
					lossc = nn.functional.mse_loss(c, c2)
				else:
					# run the decoder and feature extractor
					# compute the distance of the features from target
					z0 = denormaliseZ(pred)
					labels = decodeFeatures(vaeTrainer, z0, f)
					labels = (labels - cMin)/cSpan
	
					if oneHot:
						c2 = torch.tensor([[(oneHotDecode(i)-cMin)/cSpan] for i in c2])
	
					# compute the true perceptual loss
					sumDistance = 0
					lenBatch = len(z1)
					for i in range(lenBatch):
						distance = 1 if labels[i]<0 else (labels[i]-c2[i][0])**2
						sumDistance += distance
					lossc = sumDistance / lenBatch
	
					# using the ZC network, make a second estimation of lossc, this time with gradient
					if withZC:
						lossp = nn.functional.mse_loss(zc(pred), c2)
						with torch.no_grad():
							lossp.fill_(lossc)
						lossc = lossp

				loss = lambda_1 * lossr + lambda_2 * lossi + lambda_3 * lossc

				loss.backward()

				opt.step()

			fLoss = getFloat(loss)
			if fLoss < bestLoss:
				bestLoss = fLoss
				print(f"New best loss={bestLoss}", flush=True)
				model.save(modelSaveFile)

			result = list(map(getFloat, [loss, lossr, lossi, lossc, bestLoss ]))
			print("Loss {} : lossr {}, lossi {}, lossc {}".format(*result), flush=True)
			print(epoch, *result, file=log, flush=True)

		log.close()

	if profile:
		# save human- and machine-readable profile
		# to view: snakeviz path/to/NeuralTransfer2.prof
		profiler.disable()
		fProf =  open(os.path.join(path, "NeuralTraversal2.prof.txt"), "w")
		stats = pstats.Stats(profiler, stream=fProf).sort_stats('cumtime')
		stats.dump_stats(os.path.join(path, "NeuralTraversal2.prof"))
		stats.print_stats()
		fProf.close()

# failThreshold - minimum required loss, otherwise retry training
# repeatEpochs - number of times to retry training
# lr - learning rate

def train_ZtoC(path, features=None, oneHot=False, lr=0.002, failThreshold=0.01, threads=None, epochs=epochs, repeatEpochs=10):
	rd = util.openSource(path, oneHot=oneHot)

	setThreads(threads)

	# which features to train?
	if features is None:
		features = range(rd.nFeatures)

	for f in features:
		print(f"Feature {f}")
		modelSaveFile = modelFileName(path, f, "ZC")

		# read training data, with limited length if requested
		Z_1, C_1, Z_s, C_s = rd.read(f)

		print("z", Z_1.shape, "c", C_1.shape)
	
		N = C_1.shape[0]
		width = Z_1.shape[1]
		cWidth = C_1.shape[1]

		log = open(os.path.join(path,f"train_zc_{f}.csv"), "w")

		done = False
		seq = 0
		bestLoss = 1e6
		while not done:
			model = ZtoC(width, cWidth, scale=4)
			showModel(model)
	
			# prepare to optimise trainable parameters
			opt=torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=lr)
	
			# create a DataLoader for requested batch size
			dataset = SentDataset(Z_1, C_1, C_s, Z_s)
			loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	
			for epoch in tqdm(range(epochs)):
				for (z1, c1, c2), z2 in loader:
					opt.zero_grad()
	
					p = model(z1)
					loss = nn.functional.mse_loss(p, c1)
					loss.backward()
	
					p = model(z2)
					loss = nn.functional.mse_loss(p, c2)
					loss.backward()
	
					opt.step()
	
				fLoss = getFloat(loss)
				if fLoss < bestLoss:
					bestLoss = fLoss
					print(f"New best loss={bestLoss}", flush=True)
					model.save(modelSaveFile)
	
				result = list(map(getFloat, [loss, bestLoss ]))
				print("Loss {} (best {})".format(*result), flush=True)
				print(seq, epoch, *result, file=log, flush=True)

				seq += 1

			done = bestLoss < failThreshold or seq >= repeatEpochs*epochs

		log.close()

# predict - use SeNT model to predict outputs for testing data
# Results (for each feature <f>):
# -> predicted Z values for testing data are saved to "sp_{f}.csv"
# -> Z2 which is a known alternative solution is saved to "z2_{f}.csv"
# -> decoded features are saved to "cp_{f}.csv"
# 
# path - path to SeNT model directory
#
# oneHot - if defined, use one-hot encoding for categorical feature
#
# features - which features to predict. Default (None) is all features.
#
# decode - if true, decode the predicted position and determine features


def predict(path, oneHot=False, features=None, decode=True, threads=None, split=None):
	setThreads(threads)

	rd = util.openSource(path, oneHot=oneHot, train=False)
	header = ",".join(rd.names)

	if features is None:
		features = range(rd.nFeatures)

	if decode:
		source = rd.config['source']
		vaeTrainer = dmutil.openModel(source, split=split)

	for f in features:
		print(f"Feature {f}")
		# read Sent Model
		model = SentNet.load(modelFileName(path, f))

		# read test data
		z1, c1, z2, c2 = rd.read(f, require=4)

		# predict new Z
		z1 = torch.tensor(z1, dtype=torch.float32)
		c1 = torch.tensor(c1, dtype=torch.float32)
		c2 = torch.tensor(c2, dtype=torch.float32)
		with torch.no_grad():
			z = model(z1, c1, c2)

		# save predicted Z
		z = util.denormalise1(z.numpy(), rd.config)
		util.saveData(path, f"sp_{f}.csv", z)

		# save reference Z2
		# This Z2 will produce the desired C2, but may be different from Z
		z2 = util.denormalise1(z2, rd.config)
		util.saveData(path, f"z2_{f}.csv", z2)

		if decode:
			print("Decoding...")
			z = torch.tensor(z, dtype=torch.float32)
			labels = decodeFeatures(vaeTrainer, z)
			np.savetxt(os.path.join(path,f"cp_{f}.csv"),
				labels, delimiter=',',header=header, comments="", fmt='%d')

fire.Fire({
	'train' : train_neural_traversal,
	'trainzc' : train_ZtoC,
	'predict' : predict
})
