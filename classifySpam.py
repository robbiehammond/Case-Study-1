# -*- coding: utf-8 -*-
"""
Use XGBoost Gradient Boosted Ensemble model to classify spam emails

@authors: Prateek Dullur and Robbie Hammond
Used some code from Kevin S. Xu and public libraries: sklearn, numpy, xgboost, imblearn
Requires installation of xgboost and imbalanced-learn to run

References
-------------
XGBoost - 
Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). New York, NY, USA: ACM. https://doi.org/10.1145/2939672.2939785

SMOTE -
N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," Journal of artificial intelligence research, 321-357, 2002
"""

import xgboost as xgb   
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE 

def aucCV(features,labels, model):
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')  
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    rus = SMOTE(random_state=42)    # fit predictor and target variable
    trainFeatures, trainLabels = rus.fit_resample(trainFeatures, trainLabels)
    clf = xgb.XGBClassifier(colsample_bytree=0.73, max_depth=4, eta=0.1, max_bin=12, n_estimators=100, objective='binary:logistic') 
    clf.fit(trainFeatures, trainLabels)
    testOutputs = clf.predict_proba(testFeatures)[:,1]    
    return testOutputs