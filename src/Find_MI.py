##############################################################
# Calculating the Mutual Information between Z (latent code) 
# dimensions and c (an attribute)

# INPUT: [Latent1] [Latent2] [Att1] [Att2] [the top N dimensions to be found]
# OUTPUT: Save the N Correlated and (512-N) Uncorrelated indices in two files. 

##############################################################

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.feature_selection
import util
import fire

##################################################################### 
def find_mi(source, count, miType, limit=None, features=None):
	assert(miType in [ "MI", "XC" ])
	rd = util.openSource(source)

	if features is None:
		features = range(rd.nFeatures)
	
	for f in features:
		print(f"Feature {f}")
		Z, C, _, _ = rd.read(f)
		if limit is not None:
			Z=Z[:limit]
			C=C[:limit]

		print("Z", Z.shape, "C", C.shape)
	
		if miType == "MI":
			# compute mutual information
			MI = []
			for i in tqdm(range(len(Z[0]))):
				mi_ = sklearn.feature_selection.mutual_info_regression(Z[:,i].reshape(-1,1),
						C.ravel(), 
						discrete_features = False,
						n_neighbors = 10)[0]
				MI.append(mi_)
		else:
			# compute cross-correlation
			MI = []
			for i in tqdm(range(len(Z[0]))):
				z = Z[:,i]
				C = C.ravel()
				mi_ = np.corrcoef(z, C)
				MI.append(abs(mi_[0,1]))
		
	#####################################################################
		X_ = np.arange(0,len(Z[0]))
		plt.title(f"Feature {f} {rd.names[f]}")
		print("MI", MI)
		plt.bar(X_,MI)
		plt.xlabel("Dimensions",fontsize=20)
		plt.ylabel(f"{miType} between att.",fontsize=20)
		if Z.shape[1] > 500:
			plt.xticks([100,200,300,400,500])
		plt.savefig(f"{source}/{miType}_{f}.png",bbox_inches='tight')
		plt.clf()
	
	#####################################################################
		sm = np.array(MI).argsort()[-count:][::-1]
		np.savetxt(f"{source}/corr_index_{f}.txt",sm)
		np.savetxt(f"{source}/uncorr_index_{f}.txt",list(set(X_)-set(sm)))

fire.Fire(find_mi)
