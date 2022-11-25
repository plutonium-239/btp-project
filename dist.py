from scipy import stats
import numpy as np
import pandas as pd

def stats_from_dist(sieves, probs, name):
	assert len(sieves) == len(probs)
	sieves = np.array(sieves)
	probs = np.array(probs)/100.0
	dist = stats.rv_discrete(name=name, values=(sieves, probs))
	l = [*dist.stats(moments='mvsk'), dist.moment(1), dist.moment(2), dist.median(), *dist.interval(0.5), *dist.interval(0.95)]
	print(l)
	df.loc[name] = l

dists = {
	# 'gaussian 0 d=0.4mm': [[1.095,0.92,0.7705,0.6005,0.425], [9,23,35,25,8]],
	'gaussian 0 d=0.4mm': [[0.55,0.475,0.4025,0.3275,0.25], [15,23,36,16,10]],
	'gaussian 1 d=0.5mm': [[4.5,3.5,2.75,1.875,0.94,0.49,0.225,0.05], [1,1,5,18,47,16,9,3]],
	'gaussian 2 d=0.5mm': [[4.5,3.5,2.75,1.875,0.94,0.49,0.225,0.05], [1,1,5,15,25,42,9,2]],
	'uniform 0 d=0.4mm': [[0.55,0.475,0.4025,0.3275,0.25], [27,23,18,17,15]],
	'uniform 1 d=1.0mm': [[2.75,2.05,1.425,0.94,0.49], [21,21,23,20,15]],
	'binary 0 d=0.4mm': [[0.55,0.4275,0.3275], [48,0,52]],
	'binary 1 d=1.3mm': [[2.05,1.425,0.94], [50,0,50]],
	'narrow 0 d=0.4mm': [[0.4], [100]],
	'narrow 1 d=0.5mm': [[0.5], [100]],
	'narrow 2 d=1.0mm': [[1.0], [100]],
	'narrow 3 d=1.3mm': [[1.3], [100]],
}

df = pd.DataFrame(columns=['name', 'mean', 'variance', 'skew', 'kurtosis', '1st order moment', '2nd order moment',
	'median', '50% interval low', '50% interval high', '95% interval low', '95% interval high']).set_index('name')

for i in dists:
	stats_from_dist(dists[i][0], dists[i][1], i)
	# f = f"mean: {s[0]:.4f}\tvariance: {s[1]:.4f}\tskew: {s[2]:.4f}\tkurtosis: {s[3]:.4f}"
	# f_silent = f"{s[0][0]:.4f}\t{s[0][1]:.4f}\t{s[0][2]:.4f}\t{s[0][3]:.4f}\t{s[1]:.4f}\t{s[2]:.4f}\t{s[3]:.4f}\t"

	# print(f_silent)
	# print(f_silent, s[-2:])
print(df)
df.to_csv('distribution_params.csv')