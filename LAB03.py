#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:12:48 2023

@author: charles-arthurpacton
"""

########### SVM for classification ##################

from sklearn.datasets import load_iris
iris=load_iris()
iris.feature_names
print(iris.feature_names)
print(iris.data[0:5,:])
print(iris.target[0:5])
#print(iris.data)

from sklearn.model_selection import train_test_split
X=iris.data[iris.target!=2,0:2]
y=iris.target[iris.target!=2]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(X_test.shape)

from sklearn.svm import SVC
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()
SVMmodel.score(X_test,y_test)

######


import matplotlib.pyplot as plt 
plt.scatter(X[y==0,0],X[y==0,1],color='green')   
#the plt.scatter() function is used twice to plot the points according to their class.
# all points which the class is equal to 0
plt.scatter(X[y==1,0],X[y==1,1],color='blue')
# all points which the class is equal to 1
plt.scatter(X[y==2,0],X[y==2,1],color='cyan')


##############
from sklearn.svm import SVC
SVMmodel=SVC(kernel='linear')
SVMmodel.fit(X_train,y_train)
SVMmodel.get_params()

supvectors=SVMmodel.support_vectors_
print(supvectors.shape)
print(X_train.shape)
print(y_train.shape)


print(supvectors)
plt.scatter(X_train[y_train==0,0],X_train[y_train==0,1],color='green') 
plt.scatter(X_train[y_train==1,0],X_train[y_train==1,1],color='blue') 
plt.scatter(X_train[y_train==2,0],X_train[y_train==2,1],color='cyan') 
plt.scatter(supvectors[:,0],supvectors[:,1],color='red',marker='+',s=50)

W=SVMmodel.coef_
b=SVMmodel.intercept_
import numpy as np
xgr=np.linspace(min(X[:,0]),max(X[:,0]),100)

print(W[:,0])
print(W[:,1])
print(b)
ygr=-W[:,0]/W[:,1]*xgr-b/W[:,1]
plt.scatter(xgr,ygr)

########### anomalie detection







