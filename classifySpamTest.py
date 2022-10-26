# -*- coding: utf-8 -*-
"""
Use XGBoost Gradient Boosted Ensemble model to classify spam emails

@authors: Prateek Dullur and Robbie Hammond
Used some code from Kevin S. Xu and public libraries: sklearn, numpy, xgboost
Requires pip installation of xgboost to run

"""

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb   #Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939785
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
import multiprocessing


def aucCV(features,labels, model):
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')  
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures, model):
    model.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]    
    return testOutputs

  