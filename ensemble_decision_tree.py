import numpy as np
from sklearn import svm, tree
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier


algos = ['ball_tree', 'kd_tree', 'brute']
weights = ['uniform', 'distance']
n_neighbors = range(3, 30 + 1, 1)


raw_data = np.loadtxt('spamTrain1.csv', delimiter=',')
shuffleIndex = np.arange(np.shape(raw_data)[0])
np.random.shuffle(shuffleIndex)
data = raw_data[shuffleIndex, :]
X = data[:, 0:30]
y = data[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

max_score = 0

def train_SVM_model():
    global max_score
    for i in range(3, 20 + 1, 1):
        clf1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=i, random_state=2)
        clf1.fit(X_train, y_train)
        for j in range(3, 20 + 1, 1):
            clf2 = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=j, random_state=1)
            clf2.fit(X_train, y_train)
            for k in range(3, 20 + 1, 1):
                clf3 = tree.DecisionTreeClassifier(criterion='log_loss', splitter='best', max_depth=k, random_state=0)
                clf3.fit(X_train, y_train)
                eclf = VotingClassifier(estimators=[('dt1', clf1), ('dt2', clf2), ('dt3', clf3)], voting='hard')
                eclf.fit(X_train, y_train)
                score = roc_auc_score(y_test, eclf.predict(X_test))
                if score > max_score:
                    max_score = score
                    print("score: {}, gini depth: {}, entropy depth: {}, log loss depth: {}".format(score, i, j, k))

    score = roc_auc_score(y_test, eclf.predict(X_test))



def main():
    if __name__ == '__main__':
        train_SVM_model()
main()


