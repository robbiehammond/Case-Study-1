import numpy as np

def predictTest(trainFeatures, trainLabels, testFeatures):
    pass

if __name__ == "__main__":
    raw_data = np.loadtxt('spamTrain1.csv', delimiter=',')
    X = raw_data[:, 0:30]
    y = raw_data[:, 30]
    
    # insert logic to train model here