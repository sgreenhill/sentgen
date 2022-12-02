import numpy as np
import fire
import util
import math
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pdb
from sklearn.metrics import r2_score
from scipy.interpolate import make_interp_spline

figSize = (8.27, 11.69)
#figSize=(4,6)

def window(sequence, block):
	i = 0
	while i < len(sequence):
		yield sequence[i:i+block]
		i += block

def computeScatter(x, y, scale):
	points = {}
	for i in range(len(x)):
		point = (x[i], y[i])
		
		if point not in points:
			points[point] = 0
		points[point] += 1

	rx, ry, rs = [], [], []
	for p in points:
		rx.append(p[0])
		ry.append(p[1])
		rs.append(points[p])

	sMax = max(rs)
	rs = [scale * s/sMax for s in rs]
	return rx, ry, rs


def getCommonTitle(path):
	title = path[0]
	for p in path[1:]:
		for i in range(min(len(p), len(title))):
			if title[i] != p[i]:
				title = title[0:i]
				break
		title = title[:i+1]
	return title

def expandTitle(t):
	count = 0
	title = ''
	for c in t:
		if c == '_':
			c = ' ' if count % 2 == 0 else '='
			count += 1
		title += c
	return title

def plotDist(i, j, xName, yName, data, pdf, data2=None, scale=400, title=""):
	if data2 is None:
		data2 = data
	x, y, s = computeScatter(data[:,i],data2[:,j], scale)
	fig, plot = plt.subplots(figsize=(figSize[0],figSize[0]))
	plot.scatter(x, y, s=s, c=s, cmap='jet')
	plot.set_title(f"{title} Normalised Distribution {yName} vs {xName}")
	plot.set_xlabel(xName)
	plot.set_ylabel(yName)
	pdf.savefig(fig)

def plotSpace(path, output, title=""):
	data = np.genfromtxt(path, delimiter=',')
	print("data", data.shape)
	pdf = matplotlib.backends.backend_pdf.PdfPages(output)

	plotDist(1, 3, "C1", "CP", data, pdf, title=title)
	plotDist(2, 3, "C2", "CP", data, pdf, title=title)
	plotDist(1, 2, "C1", "C2", data, pdf, title=title)

	pdf.close()

def plotSource(path, output, count=None, block=0, train=False, scale=400, title=""):
	rd = util.openSource(path, train=train)
	pdf = matplotlib.backends.backend_pdf.PdfPages(output)
	for f in range(rd.nFeatures):
		C1, C2 = rd.read(f, require=2, fullC=True)
		if count is not None:
			begin = block * count
			end = begin + count
			C1 = C1[begin:end,:]
			C1 = C2[begin:end,:]
		name = rd.names[f]
		plotDist(f, f, f"C1 {name}", f"C2 {name}", C1, pdf, data2=C2, scale=scale, title=title)
	pdf.close()

def perform(*path, output='perform.pdf', block=50, predictBase='c', saveCSV=None, interpolate=False, withTitle=True, withScatter=False, categoricalMetric=False, fontsize=18):
	assert(len(path)>0)

	# get in-plot title
	title = getCommonTitle([p.lower().replace("c_50.0_","") for p in path])
	if title == '':
		withTitle = False
	else:
		title = expandTitle(title)

	# check features
	rd = [ util.openSource(p, train=False) for p in path ]
	nFeatures = rd[0].nFeatures
	fNames = rd[0].names
	fTry = list(range(nFeatures))
	nF = len(fTry)

	pdf = matplotlib.backends.backend_pdf.PdfPages(output)

	dTarget = []
	dUntarget = []
	dValid = []
	r2c = []
	r2r = []
	for f in fTry:
		name = fNames[f]
		print()
		print(f"----- Feature {f} : {name} -----")
		result = []
		deltaTarget = 0
		deltaUntarget = np.zeros((nFeatures))
		valid = 0
		for rdi in rd:
			# try:
				C1, C2 = rdi.read(f, require=2, fullC=True)
				print("C1", C1.shape, np.amin(C1), np.amax(C1))
				print("C2", C2.shape, np.amin(C2), np.amax(C2))
				CP = util.normalise2(util.loadFeatures(rdi.path, f"{predictBase}_{f}.csv"), rdi.config)
				print("CP", CP.shape, np.amin(CP), np.amax(CP))
				N = len(C1)
				assert(N == len(CP))
		
				for i in range(N):
					c1 = C1[i][f]
					c2 = C2[i][f]
					cp = CP[i][f]
		
					if cp >= 0:
						if c1 == 0:
							r = rp = 1
						else:
							r = (c2 - c1) / c1
							rp = (cp - c1) / c1
						result.append([i, c1, c2, cp, r, rp, abs(cp-c2) ]+list(abs(CP[i]-C1[i])))
						if categoricalMetric:
							if cp != c2:
								deltaTarget += 1
							for j in range(nFeatures):
								if CP[i][j] != C1[i][j]:
									deltaUntarget[j] += 1
						else:
							deltaTarget += abs(cp - c2)
							deltaUntarget += abs(CP[i]-C1[i])
						valid += 1
			# except Exception as e:
			# 	print("Exception:", e)

		if valid == 0:
			valid = 1
		dValid.append(valid/N)
		deltaTarget = deltaTarget / valid
		deltaUntarget[f] = 0
		deltaUntarget = deltaUntarget / valid

		dTarget.append(deltaTarget)
		dUntarget.append(deltaUntarget)
	
		print("deltaTarget", deltaTarget)
		print("deltaUntarget", deltaUntarget)
		util.saveList(path[0], f"fit_{f}.csv", result)

		fig, (cplot, rplot) = plt.subplots(2, 1, figsize=figSize)

		result = np.array(result)

		c_r2 = r2_score(result[:,2], result[:,3])
		r2c.append(c_r2)
		print("c:r^2", c_r2)

		r_r2 = r2_score(result[:,4], result[:,5])
		r2r.append(r_r2)
		print("r:r^2", r_r2)

		if withScatter:
			# very large scatter plots may produce damaged PDFs
			size=2
			cplot.annotate("r-squared = {:.3f}".format(c_r2), (0, max(result[:,3])))
			cplot.scatter(result[:,2], result[:,3], s=size)
			if withTitle:
				fig.suptitle(title)
			cplot.set_title(f"Normalised attribute: {name}")
			cplot.set_xlabel("C Target")
			cplot.set_ylabel("C Result")
	
			rplot.annotate("r-squared = {:.3f}".format(r_r2), (-1, 6.5))
			rplot.scatter(result[:,4], result[:,5], s=size)
			rplot.set_title(f"Relative attribute: {name}")
			rplot.set_xlabel("R Target")
			rplot.set_ylabel("R Result")
			rplot.set_xlim([-1, 7])
			rplot.set_ylim([-1, 7])
			pdf.savefig(fig)
			fig.clf()
	
		fig, cplot = plt.subplots(figsize=(figSize[0],figSize[0]))
		c_r2 = r2_score(result[:,2], result[:,3])

		fs = 14
		cplot.annotate("$R^2$ = {:.3f}".format(c_r2), (0, max(result[:,3])),fontsize=fs)
		x, y, s = computeScatter(result[:,2],result[:,3], 400)
		cplot.scatter(x, y, s=s)
		if withTitle:
			fig.suptitle(title)
		cplot.set_title(f"Normalised attribute: {name}")
		cplot.set_xlabel("C Target")
		cplot.set_ylabel("C Result")
		for item in ([cplot.title, cplot.xaxis.label, cplot.yaxis.label] + cplot.get_xticklabels() + cplot.get_yticklabels()):
			item.set_fontsize(fs)
		pdf.savefig(fig)
		fig.clf()

		fig, (p1, p2, p3) = plt.subplots(3, 1, figsize=figSize)
		result = [ x for x in result if x[4] <= 7 ]
		h = np.array(sorted(result, key = lambda x : x[4]))
		if interpolate:
			x = h[:,4]
			x = np.linspace(x.min(), x.max(), 100)
			spl = make_interp_spline(h[:,4], h[:,6], k=3)
			y = spl(x)
		else:
			x = [ np.average(h[i:i+block, 4]) for i in range(0, len(h), block) ]
			y = [ np.average(h[i:i+block, 6]) for i in range(0, len(h), block) ]

		p1.plot(x, y)
		if withTitle:
			fig.suptitle(title)
		p1.set_title(f"Deviation from desired value of {name}")
		# p1.set_xlabel("R Target")
		p1.set_xlim([-1, 7])
		p1.set_ylabel("Deviation from target")

		for ff in fTry:
			if ff != f:
				if interpolate:
					spl = make_interp_spline(h[:,4], h[:,7+ff], k=3)
					y = spl(x)
				else:
					y = [ np.average(h[i:i+block, 7+ff]) for i in range(0, len(h), block) ]
#				y = h[:,7+ff]
				p2.plot(x, y, label=fNames[ff])
		p2.set_title(f"Unintended changes in non target attributes")
		p2.set_xlabel("R Target")
		p2.set_xlim([-1, 7])
		p2.set_ylabel("Error in non-target attribute")
		p2.legend()

		table = [ [ 'delta targeted', name, deltaTarget ] ] + [ [ 'delta untargeted', fNames[i], deltaUntarget[i] ] for i in fTry if i != f ]
		p3.axis('off')
		# p3.axis('tight')
		p3.table(table, loc='center')

		pdf.savefig(fig)
		fig.clf()

	fig, (p1, p2, p3) = plt.subplots(3, 1, figsize=figSize, gridspec_kw={'height_ratios':[1, 1, 2]})
	if withTitle:
		fig.suptitle(title)
	p1.set_title(f"Error in target")
	p1.plot(dTarget)
	p1.set_xticks(range(nFeatures))
	p1.set_xticklabels(labels=fNames, rotation=45)
	p2.set_title("Proportion of targets with valid attributes")
	p2.plot(dValid)
	p2.set_xticks(range(nFeatures))
	p2.set_xticklabels(labels=fNames, rotation=45)

	pos = p3.imshow(dUntarget, vmin=0, vmax=1, cmap='jet')
	p3.set_title("Error in non-targeted attributes")
	p3.set_yticks(range(nFeatures))
	p3.set_yticklabels(labels=fNames)
	p3.set_xticks(range(nFeatures))
	p3.set_xticklabels(labels=fNames, rotation=90)
	# plt.subplots_adjust(hspace=0.2)
	fig.colorbar(pos)
	fig.tight_layout()
	pdf.savefig(fig)
	fig.clf()

	blackThreshold=0.35

	def displayVal(val):
		val = "%.2f" % val;
		return val.replace("0.00","0").replace("1.00", "1")

	if saveCSV is not None:
		with open(saveCSV, "w") as fSave:
			print(",".join(['key']+fNames), file=fSave)
			print(",".join(map(str, ['deltaTarget']+dTarget)), file=fSave);
			print(",".join(map(str, ['valid']+dValid)), file=fSave);
			print(",".join(map(str, ['c:r^2']+r2c)), file=fSave);
			print(",".join(map(str, ['r:r^2']+r2r)), file=fSave);
			for r,row in enumerate(dUntarget):
				print(",".join(map(str, ['deltaUntarget-'+fNames[r]]+list(row))), file=fSave);

	# transpose dUntarget so targeted attribute is horizontal
	dUntarget = np.array(dUntarget).transpose()

#	fig, (p, q) = plt.subplots(2, 1, figsize=(8.27, 11.69), gridspec_kw={'height_ratios':[3, 1.1]})
	fig, (p, q) = plt.subplots(2, 1, figsize=figSize, gridspec_kw={'height_ratios':[3, 1.1]})
	ppos = p.imshow(dUntarget, vmin=0, vmax=1, cmap='jet')
	n = len(dUntarget)
	for y in range(n):
		for x in range(n):
			if x != y:
				val = dUntarget[y][x]
				color = "white" if val < blackThreshold else "black"
				p.text(x, y, displayVal(val), ha="center", va="center", color=color, fontsize=fontsize)

	if withTitle:
		fig.suptitle(title)
	p.set_title(f"Error in non-targeted attributes")
	p.set_yticks(range(nFeatures))
	p.set_yticklabels(labels=fNames)
	p.tick_params(axis='x',bottom=False,labelbottom=False)
	#p.set_xticks(range(nFeatures))
	#p.set_xticklabels(labels=fNames, rotation=90)
	# p.set_xlabel("Untargeted attribute")
	#p.set_ylabel("Targeted attribute")
	p.set_ylabel("Untargeted attribute")

	qpos = q.imshow([dTarget], vmin=0, vmax=1, cmap='jet') 
	q.set_xticks(range(nFeatures))
	q.set_xticklabels(labels=fNames, rotation=90)
	q.set_xlabel("Targeted attribute")
	q.set_yticklabels([])
	q.set_yticks([])
	q.set_title(f"Error in targeted attributes")
	for x in range(n):
		val = dTarget[x]
		color = "white" if val < blackThreshold else "black"
		q.text(x, 0, displayVal(val), ha="center", va="center", color=color, fontsize=fontsize)
	# fig.colorbar(pos, ax=p, location='bottom', fraction=0.038, pad=0.20)
	# fig.subplots_adjust(right=1)
	fig.colorbar(qpos, ax=q, location='bottom', pad=0.55, aspect=30)

	#fig.tight_layout()
	plt.subplots_adjust(left=0.15,right=0.95,top=0.99, bottom=0.1, hspace=0.0)

	pdf.savefig(fig)
	fig.clf()

	pdf.close()

if __name__ == "__main__":
	fire.Fire(perform)

