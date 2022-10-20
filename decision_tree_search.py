import numpy as np
from sklearn import tree
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


criterions = ['gini', 'entropy', 'log_loss']
splitters = ['best', 'random']
max_depths = range(1, 30 + 1, 1)


raw_data = np.loadtxt('spamTrain1.csv', delimiter=',')
shuffleIndex = np.arange(np.shape(raw_data)[0])
np.random.shuffle(shuffleIndex)
data = raw_data[shuffleIndex, :]
X = data[:, 0:30]
y = data[:, 30]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

max_score = 0

def train_SVM_model(criterions, splitters, max_depths):
    global max_score



    # perform sbs
    sfs = SequentialFeatureSelector(tree.DecisionTreeClassifier(criterion=criterions, splitter=splitters, max_depth=max_depths), n_features_to_select=29, scoring='roc_auc')
    sfs.fit(X_train, y_train)
    features = sfs.get_support()

    clf = tree.DecisionTreeClassifier(criterion=criterions, splitter=splitters, max_depth=max_depths)
    new_X_train = X_train[:, features]
    new_X_test = X_test[:, features]
    clf.fit(new_X_train, y_train)

    score = roc_auc_score(y_test, clf.predict(new_X_test))

    print(score)
    if score > max_score:
        max_score = score
        print("criterion: {}, splitter: {}, max_depth: {}, score: {}, features selected: {}".format(criterions, splitters, max_depths, score, [i for i, x in enumerate(features) if x]))



def main():
    global cr
    for criterion in criterions:
        for splitter in splitters:
            for max_depth in max_depths:
                train_SVM_model(criterion , splitter, max_depth)

if __name__ == '__main__':
    main()


