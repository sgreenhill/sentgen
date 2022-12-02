import time
import util
import numpy as np
import fire
from tqdm import tqdm
import os
import random

# ---- initialise repository ----
# - compute ranges of data for normalisation

# faster pair generation which avoids repeats

def gen(path, threshold=0.01, limit=None, suffix=""):
	config = util.readConfig(path)
	names = config['fNames']

	C = util.loadFeatures(path, f"f{suffix}.csv")
	C = util.normalise2(C, config)
	print("aMin", np.amin(C), "aMax", np.amax(C))

	# we will compute dist**2 to avoid sqrt, so compare with threshold **2
	threshold = threshold ** 2

	# number of samples
	N = C.shape[0]

	# number of features
	F = C.shape[1]

	startTime = time.time()

	counts = []
	# loop over features
	for k in range(F):
		out = open(os.path.join(path,f"p{suffix}_{k}.csv"), "w")
		count = 0
		index = list(range(N))
		if limit is not None:
			bar = tqdm(total=limit)
		else:
			bar = tqdm()

		while len(index)>0:
			i = index.pop(0)
			ci = C[i]
			for pj in range(len(index)):
				j = index[pj]
				cj = C[j]
				diff = 0
				for m in range(F):
					if k != m:
						d = ci[m] - cj[m]
						diff += d * d
				if diff < threshold:
					count += 1
					bar.update(1)
					out.write(f"{i},{j},{diff}\n")
					del index[pj]
					if limit is not None and count >= limit:
						# stop iteration when limit is reached
					 	index = []
					break
		bar.close()
		counts.append(count)
		out.close()
	endTime = time.time()

	for k in range(F):
		print(k, names[k], counts[k])
	total = sum(counts)
	print('total', total)

	duration = endTime - startTime
	rate = total / duration
	print(f"{total} pairs in {duration} seconds")
	print(f"{rate} pairs per second")

# pair generation for categorical data

def gencat(path, threshold=1, limit=None, suffix=""):
	config = util.readConfig(path)
	names = config['fNames']

	C = util.loadFeatures(path, f"f{suffix}.csv")
	print("aMin", np.amin(C), "aMax", np.amax(C))

	# number of samples
	N = C.shape[0]

	# number of features
	F = C.shape[1]

	count = [ 0 ] * F
	active = list(range(F))

	startTime = time.time()
	for k in range (F):
		attrs = [ i for i in range(F) if i != k ]
		print("attrs", attrs)
		out = open(os.path.join(path,f"pc{suffix}_{k}.csv"), "w")
		if limit is not None:
			bar = tqdm(total=limit)
		else:
			bar = tqdm()

		try:
			for i in range(N):
				ci = C[i]
				j = 0
				while j < i:
					cj = C[j]

					diff = 0
					for m in attrs:
						if ci[m] != cj[m]:
							diff += 1
					if diff < threshold:
						count[k] += 1
						bar.update(1)
						out.write(f"{i},{j},{diff}\n")
						if limit is not None and count[k] >= limit:
							print("Limit reached")
							raise Break()
					j += 1
		except Break:
			pass
		out.close()
		bar.close()

	endTime = time.time()

	for k in range(F):
		print(k, names[k], count[k])
	total = sum(count)
	print('total', total)

	duration = endTime - startTime
	rate = total / duration
	print(f"{total} pairs in {duration} seconds")
	print(f"{rate} pairs per second")

# output all features from repository into CSV file

def getf(path, fileName, suffix=None):
	rd = util.openSource(path, suffix)
	F = rd.getF()
	with open(fileName, "w") as out:
		print(F.shape[0], F.shape[1], file=out)
		for i in range(F.shape[0]):
			print(" ".join(list(map(str, F[i]))), file=out)

# import features from CSV -> NPZ format

def importf(path, saveFile, limit=None):
	i = 0
	pairs = []
	while True:
		fileName = os.path.join(path, f"p_{i}.csv")
		if not os.path.exists(fileName):
			break
		inPairs = np.genfromtxt(fileName, delimiter=',', dtype=int)
		inPairs = inPairs[:,:2]
		if limit is not None:
			inPairs = inPairs[:limit]
		print(f"{fileName} -> {inPairs.shape}")
		pairs.append(inPairs.tolist())
		i += 1
	np.savez(saveFile, *pairs)

def printz(path, limit=None):
	rd = util.openSource(path, debug=False)
	z = rd.getZ()
	if limit is not None:
		z = z[:limit]
	for i in range(len(z)):
		print(",".join(map(str, z[i])))

def printf(path, limit=None, denormalise=False):
	rd = util.openSource(path, debug=False)
	if denormalise:
		f = rd.getRawF()
	else:
		f = rd.getF()
	if limit is not None:
		f = f[:limit]
	for i in range(len(f)):
		print(",".join(map(str, f[i])))

fire.Fire({
	'gen' : gen,
	'gencat' : gencat,
	'getf' : getf,
	'importf' : importf,
	'init2' : util.init2,
	'test' : util.testReader,
	'printz' : printz,
	'printf' : printf
})
