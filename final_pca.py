print('hello world')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# load dataset

data = pd.read_csv('data/normalised_RNAseq_noNa.csv', header = 0, index_col = 0)


X = data.drop('class', axis = 1)
y = data['class']

# PCA


pca = PCA(n_components= 50)

pca.fit_transform(X)


# to recieve the most important gene for each principle component list comprehension was used as seen below:

most_important = [np.abs(pca.components_[i]).argmax() for i in range(50)] # pca.components_ is an array of components and underlying features
PC_index = [1,2,8,9,15,16,21,22,26,27,29,31,32,33,34,36,40,44,48]

most_most_important = [most_important[i] for i in PC_index]
final_list = [f'PC-{PC_index[i]+1}: {data.columns[most_most_important][i]}' for i in range(0,len(PC_index))]
print(final_list)

ascend = np.sort(pca.components_[34]) # sort PC 35 
descend = ascend[::-1]  # reverse it, as now the smallest are first
top200_pos = descend[:200] # get the first 200 variances
# print(top200_pos)

# To find the gene names of the top 200 variances underlying PC 35 np.where is used

# then the location of these genes is matched with the name of the genes in the rows in order to get gene names. This is output in a list and printed

pos_genes = [data.columns[i] for i in np.where(pca.components_[34] >= top200_pos[-1])[0]] 
print(pos_genes)
print('this script is finished')