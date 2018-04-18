import pandas as pd 
import os
import sys
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
correlation_lib = '/Users/lishengsun/Dropbox/AutoML16_simulations/Upload_to_git/Plots/util/'
sys.path.insert(0, correlation_lib)
from correlations import *
from scipy.cluster.hierarchy import dendrogram, linkage

matrix_df = pd.DataFrame.from_csv('/Users/lishengsun/Dropbox/Meta-RL/DATASETS/Other_datasets/results_matthias/OpenMLmatthias_rdcm_acc_svdcompleted.csv')

model_names = list(matrix_df)
task_names = list(matrix_df.index)
matrix = matrix_df.as_matrix()
Z = linkage(matrix, 'single')

plt.figure(figsize=(18, 8))
plt.title('HCA_OpenML-matthias')
plt.xlabel('model index')
plt.ylabel('distance')
dendrogram(
    Z,
    labels=model_names
    # leaf_rotation=90.,  # rotates the x axis labels
    # leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
# plt.savefig('HCA_model_AutoML+MLcomp_normalized_SVDcomplete')