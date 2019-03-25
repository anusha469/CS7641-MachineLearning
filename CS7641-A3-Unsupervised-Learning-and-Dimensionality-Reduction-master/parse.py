# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import numpy as np
import os 
import sklearn.model_selection as ms

for d in ['BASE','RP','PCA','ICA','RF']:
    n = './output/{}/'.format(d)
    if not os.path.exists(n):
        os.makedirs(n)

OUT = './output/BASE/'

wine = pd.read_csv('./data/WineQuality.txt')      
col_names_wine = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
wine.columns = col_names_wine
wine[col_names_wine] = wine[col_names_wine].astype(np.int64)


wineX = wine.drop('quality',1).copy().values
wineY = wine['quality'].copy().values
#print(wineX)

cancer = pd.read_csv('./data/breast_cancer.csv')  
cancer = pd.get_dummies(cancer, columns = ['class'], prefix = 'class')
cancer['class'] = cancer['class_2.0']
cancer.drop(['class_2.0', 'class_4.0'], axis = 1, inplace = True)
cancer = cancer.astype(np.int64)

#contra.to_hdf(OUT+'datasets.hdf','contra',complib='blosc',complevel=9)
#cancer.to_hdf(OUT+'datasets.hdf','cancer',complib='blosc',complevel=9)

cancerX = cancer.drop('class',1).copy().values
cancerY = cancer['class'].copy().values
#
wine_trgX, wine_tstX, wine_trgY, wine_tstY = ms.train_test_split(wineX, wineY, test_size=0.3, random_state=0,stratify=wineY)     
#
cancer_trgX, cancer_tstX, cancer_trgY, cancer_tstY = ms.train_test_split(cancerX, cancerY, test_size=0.3, random_state=0,stratify=cancerY)     
#
wineX = pd.DataFrame(wine_trgX)
wineY = pd.DataFrame(wine_trgY)
wineY.columns = ['quality']

wineX2 = pd.DataFrame(wine_tstX)
wineY2 = pd.DataFrame(wine_tstY)
wineY2.columns = ['quality']
#
wine1 = pd.concat([wineX,wineY],1)
wine1 = wine1.dropna(axis=1,how='all')
wine1.to_hdf(OUT+'datasets.hdf','wine',complib='blosc',complevel=9)
#
wine2 = pd.concat([wineX2,wineY2],1)
wine2 = wine2.dropna(axis=1,how='all')
wine2.to_hdf(OUT+'datasets.hdf','wine_test',complib='blosc',complevel=9)
#
#
#
cancerX = pd.DataFrame(cancer_trgX)
cancerY = pd.DataFrame(cancer_trgY)
cancerY.columns = ['class']
#
cancerX2 = pd.DataFrame(cancer_tstX)
cancerY2 = pd.DataFrame(cancer_tstY)
cancerY2.columns = ['class']
#
cancer1 = pd.concat([cancerX,cancerY],1)
cancer1 = cancer1.dropna(axis=1,how='all')
cancer1.to_hdf(OUT+'datasets.hdf','cancer',complib='blosc',complevel=9)
#
cancer2 = pd.concat([cancerX2,cancerY2],1)
cancer2 = cancer2.dropna(axis=1,how='all')
cancer2.to_hdf(OUT+'datasets.hdf','cancer_test',complib='blosc',complevel=9)