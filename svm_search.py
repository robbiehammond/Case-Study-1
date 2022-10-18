import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
c_range = range(-5, 15 + 1, 2) # 2^-5 -> 2^3
gamma_range = range(-15, 3 + 1, 2)

raw_data = np.loadtxt('spamTrain1.csv', delimiter=',')
X = raw_data[:, 0:30]
y = raw_data[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

def train_SVM_model(kernel, gamma, C):
    clf = svm.SVC(kernel=kernel, gamma=gamma, C=C)
    clf.fit(X_train, y_train)
    print('kernel: {}, gamma: {}, C: {}, score: {}'.format(kernel, gamma, C, roc_auc_score(y_test, clf.predict(X_test))))


def main():
    for kernel in kernels:
        for gamma in gamma_range:
            for C in c_range:
                train_SVM_model(kernel, 2**gamma, 2**C)

if __name__ == '__main__':
    main()


