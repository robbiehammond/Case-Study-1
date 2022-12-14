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
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import multiprocessing


def aucCV(features,labels, model):
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')  
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=multiprocessing.cpu_count() // 2)
    clf = xgb.XGBClassifier(max_depth=6, eta=0.1, max_bin=128, n_estimators=100, objective='binary:logistic')
    clf.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = clf.predict_proba(testFeatures)[:,1]    
    return testOutputs


if __name__ == "__main__":
    trainData = np.loadtxt('spamTrain1.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    np.random.seed(1)
    shuffleIndex = np.arange(np.shape(trainData)[0])
    np.random.shuffle(shuffleIndex)
    trainData = trainData[shuffleIndex,:]
    trainFeatures = trainData[:,:-1]
    trainLabels = trainData[:,-1]

    testData = np.loadtxt('spamTrain2.csv',delimiter=',')
    # Randomly shuffle rows of data set then separate labels (last column)
    np.random.seed(1)
    shuffleIndex = np.arange(np.shape(testData)[0])
    np.random.shuffle(shuffleIndex)
    testData = trainData[shuffleIndex,:]
    testFeatures = testData[:,:-1]
    testLabels = testData[:,-1]
    

    # Define xgb classifier
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs=multiprocessing.cpu_count() // 2)
    clf = xgb.XGBClassifier(max_depth=4, eta=0.1, max_bin=256, n_estimators=100, objective='binary:logistic')
    print("10-fold cross-validation mean AUC: ", np.mean(aucCV(trainFeatures,trainLabels, model=clf)))

    # Fit classifier to training data
    clf.fit(trainFeatures, trainLabels)

    # Test on testing set
    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))
    
    # Examine outputs compared to labels (code from Kevin S. Xu)
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.show()