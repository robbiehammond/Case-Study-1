# -*- coding: utf-8 -*-
"""
Use XGBoost Gradient Boosted Ensemble model to classify spam emails

@authors: Prateek Dullur and Robbie Hammond
Used some code from Kevin S. Xu and public libraries: sklearn, numpy, xgboost
Requires pip installation of xgboost to run

Only used in Report.ipynb (to run multiple classifiers)
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve


def aucCV(features,labels, model):
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')  
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures, model):
    model.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]    
    return testOutputs

def tprAtFPR(labels,outputs,desiredFPR):
    fpr,tpr,thres = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex+1]
    # Find TPR at exactly desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex+1]
    tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(desiredFPR-fprBelow) 
             + tprBelow)
    return tprAt,fpr,tpr