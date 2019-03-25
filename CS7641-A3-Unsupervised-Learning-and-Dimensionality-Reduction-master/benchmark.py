# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

out = './BASE/'
np.random.seed(0)

cancer = pd.read_hdf('./output/BASE/datasets.hdf','cancer')
cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values

wine = pd.read_hdf('./output/BASE/datasets.hdf','wine')        
wineX = wine.drop('quality',1).copy().values
wineY = wine['quality'].copy().values

wineX = StandardScaler().fit_transform(wineX)
cancerX = StandardScaler().fit_transform(cancerX)

#%% benchmarking for chart type 2

grid ={'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5, return_train_score = True)

gs.fit(wineX, wineY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'wine NN bmk.csv')


mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5, return_train_score = True)

gs.fit(cancerX, cancerY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'cancer NN bmk.csv')
#raise