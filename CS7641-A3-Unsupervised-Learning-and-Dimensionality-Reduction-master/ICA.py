

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './output/ICA/'

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

ica = FastICA(random_state=5)
kurt = {}
for dim in dims_wine:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(wineX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'wine scree.csv')


ica = FastICA(random_state=5)
kurt = {}
for dim in dims_cancer:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(cancerX)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt) 
kurt.to_csv(out+'cancer scree.csv')
#raise

#%% Data for 2

grid ={'ica__n_components':dims_wine,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)

gs.fit(wineX,wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine dim red.csv')


grid ={'ica__n_components':dims_cancer,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)       
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5,return_train_score=True)

gs.fit(cancerX,cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer dim red.csv')
#raise
#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 7
ica = FastICA(n_components=dim,random_state=10)

wineX2 = ica.fit_transform(wineX)
wine2 = pd.DataFrame(np.hstack((wineX2,np.atleast_2d(wineY).T)))
cols = list(range(wine2.shape[1]))
cols[-1] = 'class'
wine2.columns = cols
wine2.to_hdf(out+'datasets.hdf','wine',complib='blosc',complevel=9)

dim = 2
ica = FastICA(n_components=dim,random_state=10)
cancerX2 = ica.fit_transform(cancerX)
cancer2 = pd.DataFrame(np.hstack((cancerX2,np.atleast_2d(cancerY).T)))
cols = list(range(cancer2.shape[1]))
cols[-1] = 'class'
cancer2.columns = cols
cancer2.to_hdf(out+'datasets.hdf','cancer',complib='blosc',complevel=9)