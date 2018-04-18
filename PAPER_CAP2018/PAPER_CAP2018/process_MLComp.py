import pandas as pd
import numpy as np


def densify(X_df, num_row, num_col):
	"""
	densify a matrix X (represented in DataFrame) to have shape of (num_row, num_col)
	
	save_to = path to where save the densified matrix csv file
	"""
	if X_df.shape[0] < num_row or X_df.shape[1] < num_col:
		raise ValueError('Could not densify matrix of shape (%i, %i) to shape(%i, %i)' %(X_df.shape[0], X_df.shape[1], num_row, num_col))


	num_notNaN = sum(X_df.apply(lambda x: x.count(), axis=1))
	sparsity = float(X_df.size - num_notNaN)/X_df.size
	print 'sparsity before densifying: ', sparsity

	df_sort_dense_row = X_df.iloc[X_df.isnull().sum(axis=1).mul(1).argsort()] #first sort row
	sort_dense_col = df_sort_dense_row.isnull().sum(axis=0).sort_values().index #then sort col
	df_sort_dense = df_sort_dense_row[sort_dense_col] # X_df sorted such that upper-left is densest
	df_dense = df_sort_dense.iloc[:num_row,:num_col]
	print df_dense
	mat_dense = df_dense.as_matrix()
	num_notNaN_dense = sum(df_dense.apply(lambda x: x.count(), axis=1))
	sparsity_dense = float(df_dense.size-num_notNaN_dense)/df_dense.size
	
	print 'sparsity after densifying: ', sparsity_dense
	return df_dense, mat_dense, sparsity_dense


if __name__ == '__main__':
	X_df = pd.DataFrame.from_csv('DATASETS/ORIGINAL_DATA/MLComp/mlcomp_data.csv')
	num_row=300
	num_col=10
	df_dense, mat_dense, sparsity_dense = densify(X_df, num_row, num_col)
	df_dense.to_csv('DATASETS/STUDIED_DATA/MLComp/mlcomp_data.csv')
	np.savetxt('DATASETS/STUDIED_DATA/MLComp/mlcomp.data', mat_dense, fmt='%.3f')

	with open('DATASETS/STUDIED_DATA/MLComp/mlcomp_sample.name', 'w') as f_sample:
		for item in list(df_dense.index):
			f_sample.write(str(item)+'\n')

	with open('DATASETS/STUDIED_DATA/MLComp/mlcomp_feat.name', 'w') as f_feat:
		for item in list(df_dense.columns):
			f_feat.write(str(item)+'\n')

	# df_dense.to_csv('Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data_dense%ids%im.csv'%(num_row,num_col))
	# mat_dense = df_dense.as_matrix()
	# np.savetxt('Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data_dense%ids%im.data'%(num_row,num_col), mat_dense, fmt='%.3f')





