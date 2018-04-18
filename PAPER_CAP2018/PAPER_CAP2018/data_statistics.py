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


	
	


if __name__ == '__main__':
	original_data_folder = '/Users/lishengsun/Dropbox/Meta-RL/DATASETS/Other_datasets/ORIGINAL_DATA/MLComp'
	X = np.loadtxt(os.path.join(original_data_folder, 'mlcomp.data'))
	rank1 = rank(X)
	XX = np.eye(4)
	rank2 = rank(XX)
	print rank2
