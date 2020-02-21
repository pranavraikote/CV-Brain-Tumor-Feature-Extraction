# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 23:33:13 2020

@author: Pranav
"""

import pandas as pd
import numpy as np
from sklearn import metrics

data = pd.read_csv('New_Dataset.csv', delimiter=',')

positive = data.loc[data['Class']==1]
negative = data.loc[data['Class']==0]
positive = positive[53:80]

cleaned_data = pd.concat([positive, negative])
#cleaned_data = cleaned_data.drop(columns=['Eccentricity'])

X = cleaned_data.iloc[:,0:7]
Y = cleaned_data.iloc[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y)

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB().fit(xtrain,ytrain)
pred_MNB = MNB.predict(xtest)
acc_MNB = metrics.accuracy_score(ytest, pred_MNB)*100

print('1. Using Naive Bayes Method')
print('Accuracy - {}'.format(acc_MNB))
print('Recall - {}'.format(metrics.recall_score(ytest, pred_MNB)))
print('Precision Score - {}'.format(metrics.precision_score(ytest, pred_MNB)))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest, pred_MNB))
print('\n')

#For real time predictions
#Input to this function can be a 1-D Array of 29 parameters
#ex = np.array(cleaned_data.iloc[0]) or np.array(cleaned_data.iloc[500])
#ex = ex[0:7]
def new_predict(i):
    a = np.array([i])
    pred = MNB.predict(a)
    result = round(pred[0])
    if result==0:
        print('Brain Tumor - Negative')
    else:
        print('Brain Tumor - Positive')

ex = np.array(cleaned_data.iloc[0])
ex = ex[0:7]
new_predict(ex)