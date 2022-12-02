import fire
import time
import asyncio
import os
import sys
import util

semaphore = 1

def error(message):
	sys.exit(message)

# Asynchronously execute a command and wait for result
async def run(cmd, prefix=""):
	cmd = prefix + cmd
	print(">", cmd, flush=True)
	proc = await asyncio.create_subprocess_shell(cmd)
	await proc.communicate()
	print("result", proc.returncode, cmd, flush=True)

# Perform SeNT-Gen training per feature
# 1) Determine relevant dimensions via mutual information
# 2) Train Z-to-C network used for loss_c gradient
# 3) Train SeNT-Gen network
# 4) Predict Z for testing data

async def sentgen(dir, feature, s):
	async with semaphore:
		print(f"SeNT-Gen Starting {dir} {feature}")

		# must set OMP_NUM_THREADS otherwise OpenMP libraries use all available cores in every Python instance
		env = f"env OMP_NUM_THREADS={s.threads}"
		# handle a single feature, rather than processing all features
		features = f"--features {feature},"
		log = f">> {dir}/run-{feature}.log 2>&1" if s.log else ''

		if not s.sequentialMI:
			await run(f"{env} python -m Find_MI {dir} {s.gamma} MI {features} {log}")
		await run(f"{env} python -m NeuralTraversal2 trainzc {dir} --threads {s.threads} {features} {s.trainZCArgs} {log}")
		await run(f"{env} python -m NeuralTraversal2 train {dir} {s.lambda1} {s.lambda2} {s.lambda3} --split {s.split} --threads {s.threads} {features} {s.trainArgs} {log}")
		await run(f"{env} python -m NeuralTraversal2 predict {dir} --split {s.split} --threads {s.threads} {features} {s.predictArgs} {log}")

		print(f"SeNT-Gen Finished {dir} {feature}")

# Create training/testing data set
# In parallel, train/test all features
# Based on the chosen random seed r={0,1,2}, use pre-computed pairs

async def task(path, s):
	print("task", path)
	base = os.path.splitext(os.path.basename(path))[0]
	parts = base.split("_")
	if parts[0] == 'DMelodiesVAE':
		parts = parts[1:]
	if parts[0].lower() == 'rnn':
		parts = parts[1:]
	if parts[-1] == "":
		parts = parts[:-1]
	if parts[-2]=='r':
		pairs = f"{s.gen}/{s.pairs}/{parts[-1]}/pairs.npz"
		if not os.path.exists(pairs):
			error(f"Pairs file {pairs} not found")
	else:
		error("Unknown seed")
	base = "_".join(parts)

	dir = os.path.join(s.outDir, base)
	os.mkdir(dir)

	await run(f"python -m build init2 {dir} {path} {pairs} {s.buildArgs}")
	rd = util.openSource(dir, train=True, debug=True)

	if s.sequentialMI:
		env = f"env OMP_NUM_THREADS={s.threads}"
		await run(f"{env} python -m Find_MI {dir} {s.gamma} MI")

	nFeatures = rd.config['nFeatures']
	await asyncio.gather(*[sentgen(dir, i, s) for i in range(nFeatures)])
#	await run(f"python -m Perform {dir} --predictBase cp --output {dir}/perform.pdf --saveCSV {dir}/perform.csv")
	print(f"Finished {path}")

class Setting:
	buildArgs = '--trainFraction 0.8'
	trainArgs = ''
	trainZCArgs = ''
	predictArgs = ''
	outDir = 'out'
	gen = 'gen'
	pairs = 'identity'
	jobs = 1
	threads = 1
	split = '0.7,0.05,0.25'
	gamma = 3
	lambda1 = 1
	lambda2 = 1
	lambda3 = 1
	log = True
	# sequential evaluation of MI uses less memory
	sequentialMI = True

	# automatic settings from name

	def fromName(self, value):
		self.outDir = value
		if 'e' in value:
			self.buildArgs = ' --trainFraction 0.8'
		else:
			self.buildArgs = ' --trainTestPairs 32000,10000'

		# options shared with Z-to-C network
		trainArgs = ''
		if 'h' in value:
			trainArgs += ' --oneHot True'
		self.trainZCArgs = trainArgs
		self.predictArgs = trainArgs

		# options for SeNT network
		if 's' in value:
			trainArgs += ' --simulateLossC True'
		if 'c' in value:
			trainArgs += ' --withTarget True'
		if 'i' in value:
			trainArgs += ' --iscale 2'
		if 'p' in value:
			trainArgs += ' --preStages 2'
		self.trainArgs = trainArgs

# For each file, perform SeNT-Gen training and testing
# For directories, process each enclosed .npz file
# Process all targets in parallel using "jobs" concurrent processes

async def main(*files, **kwargs):
	s= Setting()
	print("kwargs", kwargs)
	for key, value in kwargs.items():
		if key == 'auto':
			s.fromName(value)
		else:
			setattr(s, key, value)

	global semaphore 
	semaphore = asyncio.Semaphore(s.jobs)

	os.mkdir(s.outDir)
	tasks = []
	for f in files:
		if os.path.exists(f):
			if os.path.isdir(f):
				for i in os.listdir(f):
					if os.path.splitext(i)[1] == '.npz':
						tasks.append(task(os.path.join(f, i), s))
			else:
				tasks.append(task(f, s))
		else:
			error(f"Error: {f} does not exist")
	await asyncio.gather(*tasks)
	print("Finished")

# def run(*files, **kwargs):
# 	asyncio.run(main(*files, **kwargs))

fire.Fire(main)

