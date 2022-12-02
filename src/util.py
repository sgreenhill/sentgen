import numpy as np
import csv
from os.path import join, exists
import json
import pdb

def loadSpace(path, extra=None):
	if extra is not None:
		path = join(path, extra)
	z = np.genfromtxt(path, delimiter=',', dtype=float)
	print("space", path, z.shape)
	return z

def loadFeatures(path, extra=None):
	if extra is not None:
		path = join(path, extra)
	c = np.genfromtxt(path, delimiter=',', skip_header=1)
	print("features", path, c.shape)
	return c

def loadFeatureNames(path, extra=None):
	if extra is not None:
		path = join(path, extra)
	return np.genfromtxt(path, delimiter=',', dtype=str, max_rows=1)

# normalise a vector S
def normalise1(c, config=None):
	if config is None:
		aMin = np.amin(c, axis=0)
		aMax = np.amax(c, axis=0)
	else:
		aMin = np.array(config['sRange'][0])
		aMax = np.array(config['sRange'][1])
	span = aMax - aMin
	return 2*(c-aMin)/span-1

def denormalise1(z, config):
	aMin = np.array(config['sRange'][0])
	aMax = np.array(config['sRange'][1])
	span = aMax - aMin
	return (z+1)*span/2+aMin

# normalise a vector F
def normalise2(c, config=None):
	if config is not None:
		aMin = np.array(config['fRange'][0])
		aMax = np.array(config['fRange'][1])
	else:
		aMin = np.amin(c, axis=0)
		aMax = np.amax(c, axis=0)
	span = aMax - aMin

#	if config is not None and 'fNorm0' in config and config['fNorm0']:
#		result =  2*(c-aMin)/span-1
#	else:
#		result = (c-aMin)/span

	result = (c-aMin)/span
	return result

def oneHotEncode(c, config):
	ranges = config['fRange']
	N = c.shape[0]
	W = c.shape[1]
	result = []
	for i in range(W):
		fMin = ranges[0][i]
		fMax = ranges[1][i]
		fCount = fMax - fMin + 1
		f = np.zeros((N, fCount))
		for j in range(N):
			f[j,c[j][i]-fMin] = 1
		result.append(f)
	return result

def oneHotEncodeFeature(c, i, config):
	ranges = config['fRange']
	N = c.shape[0]
	result = []
	fMin = ranges[0][i]
	fMax = ranges[1][i]
	fCount = fMax - fMin + 1
	f = np.zeros((N, fCount))
	for j in range(N):
		f[j,c[j]-fMin] = 1
	result.append(f)
	return result

def saveData(path, file, value):
	path = join(path, file)
	print("save", path)
	np.savetxt(path, value, delimiter=',')

def saveList(path, file, value):
	path = join(path, file)
	print("save", path)
	with open(path, "w") as f:
		for i in value:
			f.write(",".join(map(str,i))+"\n")

def cast(value):
	try:
		value = int(value)
	except:
		try:
			value = float(value)
		except:
			pass
	return value

# read CSV as a list
# cast values to strictest type: int < float < string
# this is instead of np.genfromtxt which always casts to float

def readList(path, file):
	path = join(path, file)
	print("read", path)
	result = []
	with open(path, "r") as f:
		for row in csv.reader(f):
			result.append([ cast(i) for i in row ])
	return result

# read raw lines from file

def readLines(path, file):
	path = join(path, file)
	print("read", path)
	with open(path, "r") as f:
		return f.readlines()

def readConfig(path):
	# print("readConfig", path)
	path = join(path, "config.json")
	result = {}
	with open(path, "r") as f:
		result = json.load(f)
	return result

def writeConfig(path, data):
	print("writeConfig", path)
	path = join(path, "config.json")
	with open(path, "w") as f:
		json.dump(data, f, indent=4)

# version 2 reader
# all features and latent codes in one .npz file
# all pairs for all features in one .npz file per partition
# --> faster and much more compact for large data sets

def pairsFileName(path):
	return join(path, "pairs.npz")

class Reader2:
	def __init__(self, path, oneHot=False, train=True, debug=True):
		self.path = path
		self.config = config = readConfig(path)
		self.names = config['fNames']
		self.nFeatures = len(self.names)
		self.F = None
		self.Z = None
		self.source = np.load(config['source'])
		self.pairs = None
		self.oneHot = oneHot
		self.debug = debug
		self.train = train

	def name(self, index):
		return self.config['fNames'][index]

	# return z1, c1, c2
	# only return z2 if <forPrediction>

	def getRawF(self):
		f = self.source['attributes']
		return f

	def getF(self):
		if self.F is None:
			f = self.source['attributes']
			if self.oneHot:
				self.F = oneHotEncode(f, self.config)
			else:	
				self.F = normalise2(f, self.config)
				if self.debug:
					print("F range", np.amin(self.F), np.amax(self.F))
		return self.F

	def getZ(self):
		if self.Z is None:
			z = self.source['latent_codes']
			self.Z = normalise1(z, self.config)
			if self.debug:
				print("Z range", np.amin(self.Z), np.amax(self.Z))
		return self.Z

	def getPairs(self, index):
		if self.pairs is None:
			pairsFile = self.config['pairs']
			if exists(pairsFile):
				self.pairs = np.load(pairsFile)
			else:
				raise Exception(f"No pairs file {pairsFile}")
		keys = list(self.pairs.keys())
		p = self.pairs[keys[index]]
		pRange = self.config['ttPairs'][index][ 0 if self.train else 1]
		return p[pRange[0]:pRange[1]]

	def read(self, index, require=4, fullC=False):
		pairs = self.getPairs(index)
		self.getF()

		if fullC:
			c1 = np.array([ self.F[i] for (i, j) in pairs ])
			c2 = np.array([ self.F[j] for (i, j) in pairs ])
		else:
			if self.oneHot:
				c1 = np.array([ self.F[index][i] for (i, j) in pairs ])
				c2 = np.array([ self.F[index][j] for (i, j) in pairs ])
			else:
				c1 = np.array([ self.F[i][index] for (i, j) in pairs ]).reshape(-1,1)
				c2 = np.array([ self.F[j][index] for (i, j) in pairs ]).reshape(-1,1)

		if require == 2:
			return c1, c2

		self.getZ()

		z1 = np.array([ self.Z[i] for (i, j) in pairs ])
		if require == 3:
			return z1, c1, c2

		z2 = np.array([ self.Z[j] for (i, j) in pairs ])
		if require == 4:
			return z1, c1, z2, c2
		raise Exception(f"Invalid argument require={require}")

# initialise repository, computing range of values for normalisation
# partitions defines division into subsets for train/test/etc
# write config.json

def init2(path, sourceFile, pairsFile, trainTestPairs=None, trainFraction=None):
	data = np.load(sourceFile)
	z = data['latent_codes']
	f = data['attributes']

	length = z.shape[0]
	nFeatures = f.shape[1]
	fNames = data['fNames'].tolist()
	assert(len(fNames) == nFeatures)

	# compute ranges for normalisation
	sRange = [ np.amin(z, axis=0).tolist(), np.amax(z, axis=0).tolist() ]
	fRange = [ np.amin(f, axis=0).tolist(), np.amax(f, axis=0).tolist() ]

	assert(trainTestPairs is not None or trainFraction is not None)

	ttPairs = []
	if pairsFile != 'skip':
		pairs = np.load(pairsFile)
		pairsKeys = list(pairs.keys())
		assert(len(pairsKeys) == f.shape[1])

		for i, key in enumerate(pairsKeys):
			p = pairs[key]
			nPairs = p.shape[0]
			if trainFraction is not None:
				nTrain = int(nPairs * trainFraction)
				ttPairs.append([[0, nTrain], [nTrain, nPairs]])
			elif trainTestPairs is not None:
				assert(sum(trainTestPairs) < nPairs)
				ttPairs.append([ [0, trainTestPairs[0]], [trainTestPairs[0], trainTestPairs[0]+trainTestPairs[1]] ])

	config = {
		'source' : sourceFile,
		'pairs' : pairsFile,
		'sRange' : sRange,
		'fNames' : fNames,
		'fRange' : fRange,
		'length' : length,
		'nFeatures' : nFeatures,
		'ttPairs' : ttPairs,
	}

	print("config", config)

	writeConfig(path, config)
	return config

# read repository, and test the read(.) function

def testReader(path):
	rd = openSource(path)
	config = rd.config
	fNames = config['fNames']
	print("config", config)

	for train in [True, False]:
		rd = openSource(path, train=train, debug=True)
		for i, name in enumerate(fNames):
			for fullC in [ False, True ]:
				for require in [2, 3, 4]:
					result = rd.read(i, require=require, fullC=fullC)
					print(f" {i} {name} fullC={fullC} require={require}", list(map(lambda x : x.shape, result)))

# open a repository, returning either v1 or v2 reader as requried

def openSource(path, oneHot=False, train=True, debug=True):
	# check repository version
	config = readConfig(path)
	assert('source' in config)
	return Reader2(path, oneHot, train, debug)
