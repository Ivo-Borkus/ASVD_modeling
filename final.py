print('hello world')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


data = pd.read_csv('data/normalised_RNAseq_noNa.csv', header = 0, index_col = 0)


X = data.drop('class', axis = 1)
y = data['class']


pca = PCA(n_components= 50)

X_pca = pca.fit_transform(X)

df = pd.DataFrame(X_pca)

df.to_csv('data/PCA_ivo_50.csv')

print('this script is finished')