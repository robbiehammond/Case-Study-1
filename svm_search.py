import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
c_range = range(-5, 15 + 1, 1) # 2^-5 -> 2^3
gamma_range = range(-15, 3 + 1, 1) #2^-15 -> 2^3
features_to_use = [3, 4, 7, 8, 10, 19, 27, 28, 29]

raw_data = np.loadtxt('spamTrain1.csv', delimiter=',')
shuffleIndex = np.arange(np.shape(raw_data)[0])
np.random.shuffle(shuffleIndex)
data = raw_data[shuffleIndex, :]
X = data[:, 0:30]
y = data[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

max_score = 0

def train_SVM_model(kernel, gamma, C):
    global max_score
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    score = roc_auc_score(y_test, clf.predict(X_test))
    if score > max_score:
        max_score = score
        print("kernel: {}, gamma: {}, C: {}, score: {}".format(kernel, gamma, C, score))



def main():
    for kernel in kernels:
        for gamma in gamma_range:
            for C in c_range:
                train_SVM_model(kernel, 2**gamma, 2**C)

if __name__ == '__main__':
    main()


