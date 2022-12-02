import os
import torch

from dmelodies_torch_dataloader import DMelodiesTorchDataset
from src.dmelodiesvae.dmelodies_vae import DMelodiesVAE
from src.dmelodiesvae.dmelodies_vae_trainer import DMelodiesVAETrainer, LATENT_NORMALIZATION_FACTORS
from src.dmelodiesvae.dmelodies_cnnvae import DMelodiesCNNVAE
from src.dmelodiesvae.dmelodies_cnnvae_trainer import DMelodiesCNNVAETrainer
from src.dmelodiesvae.interp_vae import InterpVAE
from src.dmelodiesvae.interp_vae_trainer import InterpVAETrainer
from src.dmelodiesvae.s2_vae import S2VAE
from src.dmelodiesvae.s2_vae_trainer import S2VAETrainer

# extract batch parameters from file name.
# example: "DMelodiesVAE_rnn_s2-VAE_b_0.2_c_50.0_g_0.1_r_0_"
# returns: {'model_type': 's2-VAE', 'net_type': 'rnn', 'b': 0.2, 'c': 50.0, 'g': 0.1, 'r': 0.0}

def nameToParams(name):
	parts = name.split("_")
	if parts[0]!='DMelodiesVAE':
		return None
	params = {
		'model_type' : parts[2],
		'net_type' : parts[1]
	}
	for i in range(3,len(parts)-1,2):
		params[parts[i]] = float(parts[i+1])
	return params

def getSplitName(split):
	return "-".join(map(lambda x : f"{int(x*100):02d}", split))

# open a dmelodies model
# model parameters are extracted from the path name
# normally, the dmelodies model will be stored in "saved_models"

def openModel(path, split=None):
	name = os.path.basename(path)
	params = nameToParams(name)
	if params is None:
		print(f"cannot parse params from {name}")
		return None

	seed = int(params['r'])
	dataset = DMelodiesTorchDataset(seed=seed)

	m = params['model_type']
	net_type = params['net_type']

	if m == 'interp-VAE':
		model = InterpVAE
		trainer = InterpVAETrainer
	elif m == 's2-VAE':
		model = S2VAE
		trainer = S2VAETrainer
	else:
		if net_type == 'cnn':
			model = DMelodiesCNNVAE
			trainer = DMelodiesCNNVAETrainer
		else:
			model = DMelodiesVAE
			trainer = DMelodiesVAETrainer

	if m == 'interp-VAE':
		vae_model = model(dataset, vae_type=net_type, num_dims=1)
	elif m == 's2-VAE':
		vae_model = model(dataset, vae_type=net_type)
	else:
		vae_model = model(dataset)

	if torch.cuda.is_available():
		vae_model.cuda()

	trainer_args = {
		'model_type': m,
		'rand': seed,
		'beta': params['b'],
		'capacity': params['c']
	}
	if 'g' in params:
		trainer_args['gamma'] = params['g']

	print("trainer_args", trainer_args)
	vae_trainer = trainer(dataset, vae_model, **trainer_args)

	# if a split is specified, load from  a specialised saved_models directory
	# this allows us to keep and compare models with different build parameters

	if split is not None:
		splitName = getSplitName(split)
		vae_model.filepath = vae_model.filepath.replace("saved_models", f"saved_models-{splitName}")

	print("filePath", vae_model.filepath)

	if os.path.exists(vae_model.filepath):
		print('Model exists. Running evaluation.')
	else:
		raise ValueError(f"Trained model doesn't exist {net_type}_{trainer_args}")
	vae_trainer.load_model()
	vae_trainer.dataset.load_dataset()

	return vae_trainer

def getDecoder(trainer):
	if isinstance(trainer, (S2VAETrainer, InterpVAETrainer)):
		return trainer.model.VAE.decoder
	else:
		return trainer.model.decoder

