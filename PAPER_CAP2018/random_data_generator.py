"""
Generate matrix of rank r, based on Isabelle's code
"""

import numpy as np 
import math
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 6]
import random

def random_data(p, n, r):
	"""
	shape of matrix = p X n
	ranke of matrix = r
	"""
	r = min(r, min(p,n))
	A = np.random.rand(p, r)
	P = np.random.rand(r, r)
	T = np.random.rand(r, n)
	S = np.dot(A, np.dot(P, T))
	return S

def add_missing_value(X, frac_missing):
	X_flat = X.flatten()
	num_missing = X.size*frac_missing
	missing_position=[]
	choose_from = range(len(X_flat))
	while len(missing_position) < num_missing:
		pos = random.choice(choose_from)
		missing_position.append(pos)
		choose_from.remove(pos)

	X_flat[missing_position] = np.nan
	X_miss = X_flat.reshape(X.shape)
	return X_miss

def plot_matrix_with_missing_value(X, frac_missing):
	from matplotlib import colors
	from matplotlib import cm
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	min_bound = np.min(X)
	max_bound = np.max(X)
	
	# fig = plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
	fig, ax = plt.subplots()

	cax = ax.imshow(X, cm.viridis, origin='lower')#, cmap=cmap, norm=norm)
	fig.colorbar(cax)
	
	# ax.set_xlim(0,X.shape[1]-1)
	# ax.set_ylim(0,X.shape[0]-1)
	# plt.xticks(range(X.shape[1]))
	# plt.yticks(range(X.shape[0]))
	plt.title('frac_missing = %f'%frac_missing)
	plt.show()

	






if __name__ == '__main__':
	r = 10
	S = random_data(40, 30, r)
	S_miss = add_missing_value(S, 0.2)
	plot_matrix_with_missing_value(S_miss)
	# from numpy.linalg import matrix_rank
	# print matrix_rank(S)
	# RMSE = []
	# for r_ in range(1,31):
	# 	u, s, vh = np.linalg.svd(S, full_matrices=False)
	# 	s_ = list(s[:r_])+[0]*(len(s)-r_)
	# 	smat_ = np.diag(s_)
	# 	print smat_.shape
	# 	S_ = np.dot(u, np.dot(smat_, vh))
	# 	rmse = math.sqrt(((S-S_)**2).sum())
	# 	RMSE.append(rmse)
	# plt.plot(range(1,31), RMSE, '.-')
	# plt.show()

