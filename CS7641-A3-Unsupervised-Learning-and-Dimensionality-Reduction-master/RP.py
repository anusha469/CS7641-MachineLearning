

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product

out = './output/RP/'
cmap = cm.get_cmap('Spectral') 

np.random.seed(0)

cancer = pd.read_hdf('./output/BASE/datasets.hdf','cancer')
cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values

wine = pd.read_hdf('./output/BASE/datasets.hdf','wine')        
wineX = wine.drop('quality',1).copy().values
wineY = wine['quality'].copy().values

wineX = StandardScaler().fit_transform(wineX)
cancerX = StandardScaler().fit_transform(cancerX)

clusters = range(2, 10)

dims_wine = range(1, 12)
dims_cancer = range(1, 10)

#raise
#%% data for 1

tmp = defaultdict(dict)
for i,dim in product(range(10),dims_wine):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(wineX), wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'wine scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims_cancer):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(cancerX), cancerX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'cancer scree1.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims_wine):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(wineX)    
    tmp[dim][i] = reconstructionError(rp, wineX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'wine scree2.csv')


tmp = defaultdict(dict)
for i,dim in product(range(10),dims_cancer):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(cancerX)  
    tmp[dim][i] = reconstructionError(rp, cancerX)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'cancer scree2.csv')

#%% Data for 2

grid ={'rp__n_components':dims_wine,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')


grid ={'rp__n_components':dims_cancer,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)           
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)

gs.fit(cancerX,cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer dim red.csv')
#raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 9
rp = SparseRandomProjection(n_components=dim,random_state=5)

wineX2 = rp.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'class'
wine2.columns = cols
wine2.to_hdf(out+'datasets.hdf','wine',complib='blosc',complevel=9)

dim = 7
rp = SparseRandomProjection(n_components=dim,random_state=5)
cancerX2 = rp.fit_transform(cancerX)
cancer2 = pd.DataFrame(np.hstack((cancerX2,np.atleast_2d(cancerY).T)))
cols = list(range(cancer2.shape[1]))
cols[-1] = 'class'
cancer2.columns = cols
cancer2.to_hdf(out+'datasets.hdf','cancer',complib='blosc',complevel=9)