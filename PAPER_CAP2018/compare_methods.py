import numpy as np
import math


import matplotlib.pyplot as plt
from APT_factorization import APT_recursiveElimination, APT_random, APT_GramSchmidt, APT_SVDbasedForward, standardize, SVD_baseline


def compare_methods(S, methods_to_compare=['SVD', 'APT_backward', 'APT_random', 'APT_GramSchmidt', 'APT_SVDbasedForward'], miss_err=True, xlim=20, random_data=False):
	"""
		Compare reconstruction error of different methods applied on matrix S
		- miss_err = True: error will be computed only on positions of missing values
		- miss_err = False: error will be computed on the whole matrix
	"""
	xlim = S.shape[1]
	if len(np.where(np.isnan(S))[0]) == 0: # X no missing values
		miss_err = False
	else: 
		missing_position = np.where(np.isnan(S)) # will be used for computing err
		median = np.nanmedian(S) # initialize with median
		S[missing_position] = median

	RMSE_dict = {}
	errorbar_dict = {}

	for method in [met for met in methods_to_compare if met != 'APT_random']:
		for r_ in range(1,S.shape[1]+1):
				# print 'run;', run
			if method == 'SVD':
				u, s, vh = np.linalg.svd(S, full_matrices=False)
				s_ = list(s[:r_])+[0]*(len(s)-r_)
				smat_ = np.diag(s_)
				S_reconstruct = np.dot(u, np.dot(smat_, vh))
			elif method == 'APT_backward':
				_,_, P_backward, A_backward, Wa, _, T_backward = APT_recursiveElimination(S, m=r_, d=S.shape[0])
				S_reconstruct = np.dot(A_backward, np.dot(P_backward, T_backward))
				
			elif method == 'APT_GramSchmidt':
				A, Wa, idx = APT_GramSchmidt(S, m=r_)
				S_reconstruct = np.dot(A, Wa)
			elif method == 'APT_SVDbasedForward':
				A, Wa, idx = APT_SVDbasedForward(S, m=r_)
				S_reconstruct = np.dot(A, Wa)
			elif method == 'SVD_baseline':
				A, Wa, idx = SVD_baseline(S, m=r_)
				S_reconstruct = np.dot(A, Wa)
			if miss_err:
				rmse = math.sqrt(((S[missing_position]-S_reconstruct[missing_position])**2).sum()/S[missing_position].size)
			else:
				rmse = math.sqrt(((S-S_reconstruct)**2).sum()/S.size)
			if method in RMSE_dict.keys():
				RMSE_dict[method].append(rmse)
				errorbar_dict[method].append(0)
			else:
				RMSE_dict[method] = [rmse]
				errorbar_dict[method] = [0]

		if not random_data:
			print len(range(1,xlim+1)), len(RMSE_dict[method]), len(errorbar_dict[method])
			plt.errorbar(range(1,xlim+1), RMSE_dict[method], yerr=errorbar_dict[method], label=method, linestyle='-', marker='o', markersize=6, barsabove=True)

	if 'APT_random' in methods_to_compare:
		for r_ in range(1,S.shape[1]+1):
			err_run = []
			for run in range(50):
				A_random, a_idx, T_random, _, P_random = APT_random(S, m=r_, d=S.shape[0]) 
				S_reconstruct = np.dot(A_random, np.dot(P_random,T_random))
				if miss_err:
					rmse = math.sqrt(((S[missing_position]-S_reconstruct[missing_position])**2).sum()/S[missing_position].size)
				else:
					rmse = math.sqrt(((S-S_reconstruct)**2).sum()/S.size)
				err_run.append(rmse)
			if 'APT_random' in RMSE_dict.keys():
				RMSE_dict['APT_random'].append(np.mean(err_run))
				errorbar_dict['APT_random'].append(np.std(err_run)/np.sqrt(50))
			else:
				RMSE_dict['APT_random'] = [np.mean(err_run)]
				errorbar_dict['APT_random'] = [np.std(err_run)/np.sqrt(50)]
		if not random_data:
			plt.errorbar(range(1,xlim+1), RMSE_dict['APT_random'], yerr=errorbar_dict['APT_random'], label='APT_random', linestyle='-', marker='o', markersize=6, barsabove=True)
	

			# 	if method == 'APT_random':
					
			

			# RMSE_dict[method] = RMSE_all_run
			# print RMSE_dict
			# if errorbar:
			# 	errorbar_dict[method] = errorbar_all_run
			# else:
			# 	errorbar_dict[method] = 0

			# # plt.plot(range(1,S.shape[1]+1), RMSE_dict[method], '.-', label=method)
			# # print method, RMSE_dict[method]
			# # print method, errorbar_dict[method]
			# plt.errorbar(range(1,S.shape[1]+1), RMSE_dict[method], yerr=errorbar_dict[method], label=method, linestyle='-', marker='o', markersize=6, barsabove=True)
	if not random_data:
		# plt.xticks(range(1,xlim+1))
		plt.xlim(0,xlim+1)	
		plt.xlabel('number of landmark algorithms')
		plt.ylabel('reconstruction error')
		plt.legend()
		plt.show()
	return RMSE_dict, errorbar_dict




