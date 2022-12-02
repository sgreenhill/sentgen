import fire
import dmutil

def splitName(split=None):
	if split is not None:
		print(dmutil.getSplitName(split))

fire.Fire(splitName)
