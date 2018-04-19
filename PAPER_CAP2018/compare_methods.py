import numpy as np
import math


import matplotlib.pyplot as plt
from APT_factorization import APT_recursiveElimination, APT_random, APT_GramSchmidt


def compare_methods(S, methods_to_compare=['SVD', 'APT_backward', 'APT_random', 'APT_GramSchmidt'], miss_err=True):
	"""
		Compare reconstruction error of different methods applied on matrix S
		- miss_err = True: error will be computed only on positions of missing values
		- miss_err = False: error will be computed on the whole matrix
	"""
	if len(np.where(np.isnan(S))[0]) == 0: # X no missing values
		miss_err = False
	else: 
		missing_position = np.where(np.isnan(S)) # will be used for computing err

	RMSE_dict = {}
	errorbar_dict = {}

	for method in methods_to_compare:
		RMSE_all_run = []
		errorbar_all_run = []
		for r_ in range(1,S.shape[1]+1):
			err_run = []
			for run in range(100):
				if method == 'SVD':
					u, s, vh = np.linalg.svd(S, full_matrices=False)
					s_ = list(s[:r_])+[0]*(len(s)-r_)
					smat_ = np.diag(s_)
					S_reconstruct = np.dot(u, np.dot(smat_, vh))
				elif method == 'APT_backward':
					_,_, P_backward, A_backward, _, _, T_backward = APT_recursiveElimination(S, m=r_, d=S.shape[0])
					S_reconstruct = np.dot(A_backward, np.dot(P_backward,T_backward))
				elif method == 'APT_random':
					A_random, _, T_random, _, P_random = APT_random(S, m=r_, d=S.shape[0]) 
					S_reconstruct = np.dot(A_random, np.dot(P_random,T_random))
				elif method == 'APT_GramSchmidt':
					A, Wa, idx = APT_GramSchmidt(S, m=r_)
					S_reconstruct = np.dot(A, Wa)
				if miss_err:
					rmse = math.sqrt(((S[missing_position]-S_reconstruct[missing_position])**2).sum()/S[missing_position].size)
				else:
					rmse = math.sqrt(((S-S_reconstruct)**2).sum()/S.size)
				err_run.append(rmse)
			RMSE_all_run.append(np.mean(err_run))
			errorbar_all_run.append(np.std(err_run))
		RMSE_dict[method] = RMSE_all_run
		errorbar_dict[method] = errorbar_all_run

		# plt.plot(range(1,S.shape[1]+1), RMSE_dict[method], '.-', label=method)
		# print method, RMSE_dict[method]
		# print method, errorbar_dict[method]
		plt.errorbar(range(1,S.shape[1]+1), RMSE_dict[method], yerr=errorbar_dict[method], label=method, linestyle='-', marker='o', markersize=6, barsabove=True)
	plt.xticks(range(1,S.shape[1]+1))
	plt.xlim(1,S.shape[1]+1)	
	plt.legend()
	plt.show()
	return RMSE_dict, errorbar_dict




