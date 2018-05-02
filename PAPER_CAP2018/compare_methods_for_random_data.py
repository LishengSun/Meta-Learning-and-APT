from random_data_generator import *
from compare_methods import compare_methods
import matplotlib.pyplot as plt
from SVD_complete_missingValues import SVD_complete_missingValues



def compare_methods_for_random_data(num_iter=50, missing=False, shape=(100, 100), rank=5, methods_to_compare=['SVD', 'APT_backward', 'APT_random']):
	RMSE_dict = {}
	errorbar_dict = {}
	RMSE_dict_iter_all_in_list = []
	for iteration in range(num_iter): # generate 50 random data
		print 'iteration: ', iteration
		if not missing:
			X_iter = random_data(shape[0], shape[1], rank)
		else:
			X_iter = random_data(shape[0], shape[1], rank) # rank 5
			frac_missing = 0.3
			S_miss = add_missing_value(X_iter, frac_missing)
			S_miss = SVD_complete_missingValues(S_miss)

		RMSE_dict_iter, _ = compare_methods(X_iter, methods_to_compare=methods_to_compare, miss_err=True, xlim=shape[1], random_data=True)
		RMSE_dict_iter_all_in_list.append(RMSE_dict_iter)

	for key in methods_to_compare:

		RMSE_dict[key] = np.average([d[key] for d in RMSE_dict_iter_all_in_list], axis=0)
		errorbar_dict[key] = np.std([d[key] for d in RMSE_dict_iter_all_in_list], axis=0) / np.sqrt(num_iter)

	for method in methods_to_compare:
		plt.errorbar(range(1,shape[1]+1), RMSE_dict[method], yerr=errorbar_dict[method], label=method, linestyle='-', marker='o', markersize=6, barsabove=True)
	plt.xlim(0,shape[1]+1)	
	plt.xlabel('number of landmark algorithms')
	plt.ylabel('reconstruction error')
	plt.legend()
	plt.title('rank=5')
	plt.show()

	return RMSE_dict, errorbar_dict