"""
Generate matrix of rank r, based on Isabelle's code
"""

import numpy as np 
import math
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
	r = 10
	S = random_data(40, 30, r)
	from numpy.linalg import matrix_rank
	print matrix_rank(S)
	RMSE = []
	for r_ in range(1,31):
		u, s, vh = np.linalg.svd(S, full_matrices=False)
		s_ = list(s[:r_])+[0]*(len(s)-r_)
		smat_ = np.diag(s_)
		print smat_.shape
		S_ = np.dot(u, np.dot(smat_, vh))
		rmse = math.sqrt(((S-S_)**2).sum())
		RMSE.append(rmse)
	plt.plot(range(1,31), RMSE, '.-')
	plt.show()

