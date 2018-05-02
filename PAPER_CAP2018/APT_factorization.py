"""
Adapt from Isabelle's Matlab code
Perform the MUD decomposition to a matrix X, such that:

X ~ M Wm = M U D
X ~ Wd D = M U D

can be used to predict performance of models applied on datasets using landmark models / datasets



Apply on MLcomp data, including some data processing adapted to that data, for application on other ML data, see APT_*.py
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import random
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from data_statistics import *
from SVD_complete_missingValues import SVD_complete_missingValues


def standardize(X):
	"""
	column wise normalization, based on Isabelle's Matlab code
	"""
	XX = np.copy(X) # no destroy X
	for iteration in range(5):
		mu_row = np.nanmean(XX, axis=1) # mean of row
		std_row = np.nanstd(XX, axis=1)
		for r in range(XX.shape[0]):
			# if sparsity(mu_row)==0 and sparsity(std_row)==0:
			XX[r,:] = (XX[r,:]-mu_row[r])/std_row[r]
			# else:
			# 	pass
		mu_col = np.nanmean(XX, axis=0) # mean of col
		std_col = np.nanstd(XX, axis=0)
		for c in range(XX.shape[1]):
			# if sparsity(mu_col) == 0 and sparsity(std_col)==0:
			XX[:,c] = (XX[:,c]-mu_col[c])/std_col[c]
			# else:
			# 	pass
		
	return XX


def APT_random(X, m, d):
	"""
	Randomly eliminate until m columns rest.
	"""
	m_idx = []
	d_idx = []
	choose_from_idx_m = range(X.shape[1])
	choose_from_idx_d = range(X.shape[0])

	while len(m_idx) < m:
		# print 'len(m_idx): ', len(m_idx)
		choice_idx_m = random.choice(choose_from_idx_m)
		# print 'choice_idx: ', choice_idx
		m_idx.append(choice_idx_m)
		# print m_idx
		# print len(m_idx)
		choose_from_idx_m.remove(choice_idx_m)
	M = X[:, m_idx]
	Wm = np.dot(np.linalg.pinv(M), X)

	while len(d_idx) < d:
		# print 'len(m_idx): ', len(m_idx)
		choice_idx_d = random.choice(choose_from_idx_d)
		# print 'choice_idx: ', choice_idx
		d_idx.append(choice_idx_d)
		# print m_idx
		# print len(m_idx)
		choose_from_idx_d.remove(choice_idx_d)
	M = X[:, m_idx]
	D = X[d_idx, :]
	U = np.dot(np.linalg.pinv(M), np.dot(X, np.linalg.pinv(D)))
	return M, m_idx, D, d_idx, U
	# return M, Wm




def APT_recursiveElimination(X_orig, m=5, d=5, l=1e-14):
	"""
	Recursively backward eliminate n models / p datasets until m / d left
	m, d = number of landmark models/datasets one wishes to keep
	l = lambda, regularization parameter
	"""
	X = np.copy(X_orig) # create copy of X_orig so this last will stay untouched
	p, n = X.shape # 1row = 1dataset, 1col = 1 model
	m = min(m, n)
	d = min(d, p)
	if l > 1:
		l = 1
	elif l < 0:
		l = 0
		
	med = np.nanmedian(X_orig) # replace missing values with matrix median
	X[np.where(np.isnan(X))] = med
	mini = np.nanmin(X) # replace -np.inf with min
	X[np.where(np.isneginf(X))] = mini
	maxi = np.nanmax(X) # replace np.inf with max
	X[np.where(np.isposinf(X))] = maxi

	#Recursive model elimination until m models are left
	m_idx = range(n)
	while len(m_idx) > m:
		M = X[:, m_idx]
		M = standardize(M)
		# try:
		# print 'sparsity of M:', sparsity(M)
		if sparsity(M) != 0:
			M, _, _, _ = SVD_complete_missingValues(M)
			# M_median = np.nanmedian(M)
			# missing_positions = np.where(np.isnan(M))
			# M[missing_positions] = M_median
		Wm = np.dot(np.linalg.pinv(M), X)
		# except LinAlgError:
		# 	exit
		wi, i = np.min(np.max(np.abs(Wm), axis=1)), np.argmin(np.max(np.abs(Wm), axis=1))
		m_idx.remove(m_idx[i])
	M = X[:, m_idx]
	
	Wm = np.dot(np.linalg.pinv(M), X_orig)
	#Recursive dataset elimination until d models are left
	d_idx = range(p)
	while len(d_idx) > d:
		D = X[d_idx, :]
		Wd = np.dot(X, np.linalg.pinv(D))
		wj, j = np.min(np.max(np.abs(Wd), axis=0)), np.argmin(np.max(np.abs(Wd), axis=0))
		d_idx.remove(d_idx[j])

	D = X[d_idx, :]
	Wd = np.dot(X, np.linalg.pinv(D))

	# compute U
	U = np.dot(np.linalg.pinv(M), Wd)

	return m_idx, d_idx, U, M, Wm, Wd, D


def APT_GramSchmidt(S, m=5):
	"""
	Use Gram-Schmidt process to select features most correlated to others
	Based on Isabelle's MATLAB code
	1) Select 'x1' best correlated with other x as first basis in initialized null space
	2) Project remaining x to N('x1') (null space of x1)
	3) Select 'x2' best correlated with other x in N('x1') as second basis
	Repeat until number of basis is m
	"""
	p, n = S.shape

	idx = [] # features already selected
	idx_ = range(n) # features not selected yet
	X = np.copy(S)
	NULL_PROJ = np.eye(p) #shape p,p

	while len(idx) < m:
		X = S[:, idx_]		
		X = np.dot(NULL_PROJ, X) # project X to null space of A
		X = standardize(X)

		# compute the feature most correlated to all other features (in null space of A)
		SS = np.dot(np.transpose(X), X) 
		
		# SS = np.corrcoef(np.transpose(X))
		SS = SS - np.diag(np.diag(SS)) # substract diagonal, no consider self correlation (corr(m1, m1))
		Stot = abs(SS)
		w, i = np.min(np.mean(np.abs(Stot), axis=1)), np.argmin(np.mean(np.abs(Stot), axis=1))
		# w, i = 0,0
		idx.append(idx_[i])
		idx_.remove(idx_[i])

		# compute the projector on the space of selected features
		# and the projector onto the null space
		A = S[:, idx]
		Wa = np.dot(np.linalg.pinv(A), S)
		
		# FEAT_PROJ = np.dot(A, np.dot(np.linalg.pinv(np.dot(np.transpose(A), A)), np.transpose(A)))
		FEAT_PROJ = np.dot(A, np.linalg.pinv(A))
		NULL_PROJ = np.eye(FEAT_PROJ.shape[0])-FEAT_PROJ
		# print idx
		# print 'Get NULL_PROJ right?', verif
	return A, Wa, idx


def APT_SVDbasedForward(S, m=5):
	"""
	
	"""
	p, n = S.shape

	idx = [] # features already selected
	idx_ = range(n) # features not selected yet
	X = np.copy(S)
	NULL_PROJ = np.eye(p) #shape p,p

	while len(idx) < m:
		# print 'choose from: ', idx_
		X = S[:, idx_]
		X = np.dot(NULL_PROJ, X) # project X to null space of A
		X = standardize(X)

		# compute the feature most correlated to all other features (in null space of A)
		U,s,V = np.linalg.svd(X, full_matrices=True)
		Proj = []
		for col in range(X.shape[1]):
			proj_col = np.dot(X[:,col], U[:, 0])# / (np.linalg.norm(X[:,col])*np.linalg.norm(U[:,0]))
			Proj.append(proj_col)

		# Dist = np.dot(np.transpose(X), U[:, 0]) # compute distance between 1e component of U and all remaining features
		# minDist = min(d for d in Dist if d > 0)
		# i = Dist.index(minDist)
		maxProj, i = np.max(Proj), np.argmax(Proj)
		# print 'Proj = ', Proj
		# print 'maxProj = ', maxProj
		# print Dist
		# print minDist
		# print Dist
		# print maxDist, i
		
		idx.append(idx_[i])
		idx_.remove(idx_[i])

		# compute the projector on the space of selected features
		# and the projector onto the null space
		A = S[:, idx]
		Wa = np.dot(np.linalg.pinv(A), S)
		
		# FEAT_PROJ = np.dot(A, np.dot(np.linalg.pinv(np.dot(np.transpose(A), A)), np.transpose(A)))
		FEAT_PROJ = np.dot(A, np.linalg.pinv(A))
		NULL_PROJ = np.ones(FEAT_PROJ.shape)-FEAT_PROJ
		verif = np.dot(FEAT_PROJ, NULL_PROJ)
		# print "idx", idx
		# print '======='
		# print 'Get NULL_PROJ right?', verif
	return A, Wa, idx


def SVD_baseline(S, m=5):
	U,s,V = np.linalg.svd(S, full_matrices=True)
	# 
	idx = []
	for Uj in range(U.shape[1]):
		corr_UjS = []
		for Sj in range(S.shape[1]):
			corr_UjSj = np.correlate(U[:,Uj], S[:,Sj])
			corr_UjS.append(corr_UjSj[0])
		idx.append(sorted(range(len(corr_UjS)), key=lambda k: corr_UjS[k])[0])

	# idx = sorted(range(len(corr_UjS)), key=lambda k: corr_UjS[k])
	print len(corr_UjS)
	# while len(idx) < m:
	# 	idx.append()
	A = S[:, idx[:m]]
	# print idx
	Wa = np.dot(np.linalg.pinv(A), S)
	return A, Wa, idx
	# idx = []
	# U,s,V = np.linalg.svd(S, full_matrices=True)

	# for Uj in range(U.shape[1]):
	# corr_UjS = []
	# 	while len(idx) < m:
	# 		for Sj in range(S.shape[1]):
	# 			corr_UjSj = np.correlate(U[:,Uj], S[:,Sj])
	# 			corr_UjS.append(corr_UjSj[0])
	# 		# print np.argmax(corr_UjS)
	# 		idx.append(np.argmax(corr_UjS))
	# print idx, len(idx)
	# A = S[:, idx]
	# Wa = np.dot(np.linalg.pinv(A), S)
	# return A, Wa, idx

def SVD_decomposition(X, r):
	"""r = reduced rank
	"""
	U,s,V = np.linalg.svd(X, full_matrices=True)
	S = np.diag(s[:r+1]) # keep only first r singular values
	U = U[:,:r+1]
	V = V[:r+1,:]
	X_reconstructed = np.dot(U, np.dot(S,V))
	return X_reconstructed



if __name__ == '__main__':
	# strategy_APT = True
	# strategy_random = False
	X_orig = pd.DataFrame.from_csv('../DATASETS/Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data.csv')
	import imp
	densify_matrix = imp.load_source('densify_matrix', '/Users/lishengsun/Dropbox/Meta-RL/DATASETS/densify_matrix.py')
	SVD_complete_missingValues = imp.load_source('SVD_complete_missingValues', '/Users/lishengsun/Dropbox/Meta-RL/DATASETS/SVD_complete_missingValues.py')
	num_row = 300
	num_col = 30
	X_df, X_before, _ = densify_matrix.densify(X_orig, num_row, num_col)
	X_df.to_csv('../DATASETS/Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data_dense%ids%im.csv'%(num_row,num_col))
	np.savetxt('../DATASETS/Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data_dense%ids%im.data'%(num_row,num_col), X_before, fmt='%.3f')
	

	X, X_return, err_final, reconstruction_err, missing_positions, col_median = SVD_complete_missingValues.SVD_complete_missingValues(X_before, normalization=True, num_iteration=500,plot_convergence=False)
	np.savetxt('../DATASETS/Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data_dense%ids%im_SVDcomp.data'%(num_row,num_col), X, fmt='%.3f')
	X_df_comp = pd.DataFrame(X, columns=X_df.columns, index=X_df.index)
	X_df_comp.to_csv('../DATASETS/Other_datasets/MLcomp_Percy/MLcomp_AutoMLFormat/mlcomp_data_dense%ids%im_SVDcomp.csv'%(num_row,num_col))
	# normalize each row between 0 and 1
	# for iteration in range(5):
	# 	for row in range(num_row):
	# 		if not np.min(X[row, :]) == np.max(X[row, :]): # for example, X[128,:] = 1
	# 			X[row, :] = (X[row, :]-np.min(X[row, :]))/(np.max(X[row, :])-np.min(X[row, :]))
	# 		else:
	# 			X[row, :] = 1
		
	y = np.zeros(X.shape) # target used for learning
	tol = 0.1
	for r in range(num_row):
		ones_r = list(np.where(np.abs(X[r,:]-np.nanmin(X[r,:]))<=tol)[0])
		# for c in ones_r:
		y[r, ones_r] = 1
	scoresTR_APT = []
	scoresTE_APT = []
	scoresTR_random = []
	scoresTE_random = []
	# scoresTR_GramSchmidt = []
	# scoresTE_GramSchmidt = []
	

	### iterate 10 times to stablize clf performance
	ALL_scoresTR_APT = [0]*num_col
	ALL_scoresTE_APT = [0]*num_col
	ALL_scoresTR_random = [0]*num_col
	ALL_scoresTE_random = [0]*num_col
	
	num_iter=5
	for iteration in range(num_iter):
		print '========================iter %i====================='%iteration
		for m in range(1,num_col+1):
			print 
			print 
			print 'm = ', m
			label_num = y.shape[1]
			iter_num = 10
			scoreTR_m_APT = 0
			scoreTE_m_APT = 0
			scoreTR_m_random = 0
			scoreTE_m_random = 0
			# scoreTR_m_GramSchmidt = 0
			# scoreTE_m_GramSchmidt = 0

			# for iteration in range(iter_num): # 10 iter to stabilize things
			train_idx, test_idx = train_test_split(X.shape[0])
			
			# if strategy_APT:
			m_idx_APT, d_idx, U, M_APT, Wm, Wd, D = APT_recursiveElimination(X, m=m, d=299) 
			M_random, m_idx_random = random_landmark_selection(X, m=m)
			# M_GramSchmidt, m_idx_GramSchmidt = APT_forwardSelection(X, m=m)

			# M_train_APT, M_train_random, M_train_GramSchmidt, y_train = M_APT[train_idx, :], M_random[train_idx, :], M_GramSchmidt[train_idx, :], y[train_idx,:]
			# M_test_APT, M_test_random, M_test_GramSchmidt, y_test = M_APT[test_idx, :], M_random[test_idx, :], M_GramSchmidt[test_idx, :], y[test_idx,:]
			M_train_APT, M_train_random, y_train = M_APT[train_idx, :], M_random[train_idx, :], y[train_idx,:]
			M_test_APT, M_test_random, y_test = M_APT[test_idx, :], M_random[test_idx, :], y[test_idx,:]


			for label in range(label_num): # learn labels separately
				y_train_label = y_train[:,label]
				# print 'label %i: number of 1: %i'%(label, len(np.where(y_train_label==1)[0]))
				y_test_label = y_test[:,label]
				clf_APT = RandomForestClassifier()
				clf_APT.fit(M_train_APT, y_train_label)
				y_pred_train_label_APT = clf_APT.predict(M_train_APT)
				scoreTR_m_APT += accuracy_score(y_train_label, y_pred_train_label_APT)
			
				y_test_label = y_test[:,label]
				y_pred_test_label_APT = clf_APT.predict(M_test_APT)
				scoreTE_m_APT += accuracy_score(y_test_label, y_pred_test_label_APT)

				# random
				clf_random = RandomForestClassifier()
				clf_random.fit(M_train_random, y_train_label)
				y_pred_train_label_random = clf_random.predict(M_train_random)
				scoreTR_m_random += accuracy_score(y_train_label, y_pred_train_label_random)
			
				y_test_label = y_test[:,label]
				y_pred_test_label_random = clf_random.predict(M_test_random)
				scoreTE_m_random += accuracy_score(y_test_label, y_pred_test_label_random)
			
				# print ('label %i: accumulated train score = %f'%(label,scoreTR_m))
				# print ('label %i: accumulated test score = %f'%(label,scoreTE_m))
			# print 'train score = ', scoreTR_m_APT/label_num
			# print 'test score = ', scoreTE_m_APT/label_num

				# GramSchmidt
				# clf_GramSchmidt = RandomForestClassifier(class_weight='balanced')
				# clf_GramSchmidt.fit(M_train_GramSchmidt, y_train_label)
				# y_pred_train_label_GramSchmidt = clf_random.predict(M_train_GramSchmidt)
				# scoreTR_m_GramSchmidt += accuracy_score(y_train_label, y_pred_train_label_GramSchmidt)
			
				# y_test_label = y_test[:,label]
				# y_pred_test_label_GramSchmidt = clf_random.predict(M_test_GramSchmidt)
				# scoreTE_m_GramSchmidt += accuracy_score(y_test_label, y_pred_test_label_GramSchmidt)
			
			scoresTR_APT.append(scoreTR_m_APT/label_num)
			scoresTE_APT.append(scoreTE_m_APT/label_num)

			scoresTR_random.append(scoreTR_m_random/label_num)
			scoresTE_random.append(scoreTE_m_random/label_num)

			# scoresTR_GramSchmidt.append(scoreTR_m_GramSchmidt/label_num)
			# scoresTE_GramSchmidt.append(scoreTE_m_GramSchmidt/label_num)
		ALL_scoresTE_random = [a+i for a,i in zip(ALL_scoresTE_random, scoresTE_random)]
		ALL_scoresTE_APT = [a+i for a,i in zip(ALL_scoresTE_APT, scoresTE_APT)]

	# plt.plot(range(1, X.shape[1]+1), scoresTR, '*-', label='Training accuracy')
	# plt.plot(range(1, X.shape[1]+1), scoresTE, '*-', label='Test accuracy')
	# plt.plot(range(1, X.shape[1]+1), scoresTR_APT, 'o-', color='red', label='APT: Training accuracy')
	stable_scoresTE_APT = [score/num_iter for score in ALL_scoresTE_APT]
	stable_scoresTE_random = [score/num_iter for score in ALL_scoresTE_random]
	plt.plot(range(1, X.shape[1]+1), stable_scoresTE_APT, '*-', color='red', label='APT')
	# plt.plot(range(1, X.shape[1]+1), scoresTE_GramSchmidt, 'o-', color='red', label='APT-forward')

	# plt.plot(range(1, X.shape[1]+1), scoresTR_random, 'o-', color='green', label='random: Training accuracy')
	plt.plot(range(1, X.shape[1]+1), stable_scoresTE_random, '*-', color='green', label='random')
	plt.legend()
	plt.xticks(range(1, X.shape[1]+1))
	plt.xlabel('a: number of landmark algorithms')
	plt.ylabel('Mean accuracy')
	plt.ylim(0.0, 1.0)
	# if strategy_APT:
	# plt.title('Top algorithms prediction (APT, MLcomp) accuracy')
	# plt.savefig('TopAlgPred-GS-MLcomp_norm')
	# plt.savefig('TopAlgPred-MLcomp_norm')
	# elif strategy_random:
	# 	plt.title('Top algorithms prediction (random, MLcomp) accuracy')
	# 	plt.savefig('TopAlgPred-random-MLcomp_norm')
	plt.show()


