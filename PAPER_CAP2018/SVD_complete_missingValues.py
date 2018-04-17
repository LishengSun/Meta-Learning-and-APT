import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random


def SVD_complete_missingValues(M, num_iteration=500, normalization=False, plot_convergence=True):
	"""
	- M: numpy array: original matrix with missing values, by default columns=datasets, rows=models
	- num_iteration: number of iterations to test the convergence
	- plot_convergence: if True, plot the reconstruction error vs. iteration
	- normalization: whether normalize M before decomposition (useful when performaces in M based on different metrics)
		if true, normalize to 0-1
	Pseudo-code:
	ground_truth_positions = positions in M where value is not missing
	M_ = replace_missing_values_by_model_median(M)
	
	while i <= num_iteration:
		U,S,V = SVD(M_)
		M_ = U*diag(S)*V
		M_[ground_truth_positions] = M[ground_truth_positions]

	"""
	# initialize M_ = M with missing values replaced by column median
	if normalization: #remove the effect of metric choice (some dataset can be more difficult just because it uses a difficult metric)
		for c in range(M.shape[1]): # for each dataset
			max_c = np.nanmax(M[:, c]) # max score for this dataset
			min_c = np.nanmin(M[:, c])
			M[:, c] = (M[:, c]-min_c) / (max_c-min_c)
	M_ = np.copy(M)
	reconstruction_err = []
	missing_positions = np.where(np.isnan(M))
	ground_truth_positions = np.where(~np.isnan(M))

	median = np.nanmedian(M)
	M_[missing_positions] = median
	i=0
	while i<num_iteration:
		# print i
		U, s, V = np.linalg.svd(M_, full_matrices=True)
		S_dim1 = U.shape[1]
		S_dim2 = V.shape[0]
		S = np.zeros((S_dim1, S_dim2))
		S[:len(s), :len(s)] = np.diag(s)
		M_ = np.dot(np.dot(U, S), V)

		convergent, rec_err = close(M, M_, ground_truth_positions)
		reconstruction_err.append(rec_err)
		# print 'Converged? ', convergent
		
		M_[ground_truth_positions] = M[ground_truth_positions]
		i += 1

	M_return = np.dot(np.dot(U, S), V)
	err_final = np.sqrt(sum((M_return[ground_truth_positions]-M[ground_truth_positions])**2))
	if plot_convergence:
		plt.plot(range(num_iteration), reconstruction_err)
		plt.xlabel('iter')
		plt.ylabel('SVD reconstruction error')
		plt.show()
		# plt.savefig('AutoML29datasets_reconstructionErr')

	return M_, M_return, err_final, reconstruction_err, missing_positions, median
	#M_ is M_return but keeping ground truth values

def random_eliminate_entry(M, num_elim = 800):
	"""
	randomly choose num_elim entries in np.array M and replace them by NaN
	"""
	M_elim = np.copy(M)
	if num_elim > M.size:
		raise ValueError('Not valid num_elim')
	else:
		possible_positions = [(row, col) for row in range(M.shape[0]) for col in range(M.shape[1])]
		elim_positions = []
		while len(elim_positions) < num_elim:
			print len(elim_positions)
			index = random.randint(0,len(possible_positions)-1)
			if possible_positions[index] not in elim_positions:
				elim_positions.append(possible_positions[index])
				M_elim[possible_positions[index][0], possible_positions[index][1]] = np.nan
	return M_elim


def close(A, B, ground_truth_positions, rtol=1e-05, atol=1e-08):
	"""
	determine if A and B are close based on ground_truth values.
	Used for determining reconstruction convergence.
	only use ground_truth_positions values in A and B
	ref: https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.allclose.html
	"""
	dist_array = (A[ground_truth_positions]-B[ground_truth_positions])**2
	dist = np.sqrt(sum(dist_array))
	if dist <= (atol+rtol*np.linalg.norm(B[ground_truth_positions])):
		return True, dist
	else:
		return False, dist


if __name__ == '__main__':
	# M_df = pd.DataFrame.from_csv('AutoML+MLcomp.csv')
	M_df = pd.DataFrame.from_csv('DATASETS/STUDIED_DATA/OpenML/OpenML_data.csv')
	M = M_df.as_matrix()
	# M_elim = random_eliminate_entry(M, num_elim = 1000)
	M_, M_return, err_final, reconstruction_err, missing_positions, col_median = SVD_complete_missingValues(M, normalization=False, num_iteration=100, plot_convergence=True)
	# M_df_completed = pd.DataFrame(M_, columns=list(M_df), index=list(M_df.index))
	# M_df_completed.to_csv('Other_datasets/results_matthias/OpenMLmatthias_svdcompleted.csv')







