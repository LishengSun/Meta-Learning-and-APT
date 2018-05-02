import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import os


def sparsity(X):
	"""input: numpy array X
		output: sparsity of X computed as (num_notnan / size)
	"""
	sparsity = float(np.count_nonzero(np.isnan(X))) / np.size(X)
	return sparsity

def num_dataset(X):
	return X.shape[0]

def num_model(X):
	return X.shape[1]

def rank(X):
	if not np.isnan(X).any():
		from numpy.linalg import matrix_rank
		return matrix_rank(X)
	else:
		# print "Matrix contains NaN"
		raise ValueError("Matrix contains NaN")

def ave_corr_rows(X):
	# from matplotlib import colors
	# from matplotlib import cm
	# from mpl_toolkits.axes_grid1 import make_axes_locatable

	corr = np.corrcoef(X, rowvar=True)
	corr = corr - np.diag(np.diag(corr))
	corr.sum()/(X.shape[0]*X.shape[0]-X.shape[0])
	
	return np.nanmean(abs(corr))
	# fig, ax = plt.subplots()

	# cax = ax.imshow(corr, cm.viridis, origin='lower')#, cmap=cmap, norm=norm)
	# fig.colorbar(cax)

	# plt.show()
def ave_corr_cols(X):
	corr = np.corrcoef(X, rowvar=False)
	corr = corr - np.diag(np.diag(corr))
	corr.sum()/(X.shape[0]*X.shape[0]-X.shape[0])
	
	return np.nanmean(abs(corr))


	
	


if __name__ == '__main__':
	original_data_folder = '/Users/lishengsun/Dropbox/Meta-RL/DATASETS/Other_datasets/ORIGINAL_DATA/MLComp'
	X = np.loadtxt(os.path.join(original_data_folder, 'mlcomp.data'))
	rank1 = rank(X)
	XX = np.eye(4)
	rank2 = rank(XX)
	print rank2
