import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = '/home/raghu/Desktop/NN'
TRAIN_FILE = DATA_PATH + '/arcene_train.data'
TRAIN_LABELS_FILE = DATA_PATH + '/arcene_train.labels'
TEST_FILE = DATA_PATH + '/arcene_valid.data'
TEST_LABELS_FILE = DATA_PATH + '/arcene_valid.labels'

file_handle = open(TRAIN_FILE)
train = np.array([list(map(int, file_handle.readline().strip().split(' '))) for _ in range(100)], dtype='float64')
train.shape
file_handle = open(TEST_FILE)
test = np.array([list(map(int, file_handle.readline().strip().split(' '))) for _ in range(100)], dtype='float64')
test.shape
file_handle = open(TRAIN_LABELS_FILE)
y_train = np.array([int(file_handle.readline().strip()) for _ in range(100)])
file_handle = open(TEST_LABELS_FILE)
y_test = np.array([int(file_handle.readline().strip()) for _ in range(100)])

#PCA K=100

pca = PCA(n_components=100)
pca.fit(train)
X_train = pca.transform(train)
X_test = pca.transform(test)

tuned_parameters = [
  {'C': [1, 10, 100, 1000, 10000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
scores_list = ['precision', 'recall']
for score in scores_list:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Final accuracy =", accuracy_score(y_test, clf.predict(X_test)))

clf_final = SVC(kernel='linear', C=1)
clf_final.fit(X_train, y_pred)

print("Number of support Vectors =", len(clf_final.support_))
print("Number of suppport vectors for each class:", clf_final.n_support_ )
print("The margin support vectors =", clf_final.dual_coef_.shape[1])
print("The non-margin support vectors = 0")

# k = 10
scaler = MinMaxScaler()
scaler.fit(train)
X_train_s = scaler.transform(train)
X_test_s = scaler.transform(test)
pca = PCA(n_components=10)
pca.fit(X_train_s)
X1_train = pca.transform(X_train_s)
X1_test = pca.transform(X_test_s)

tuned_parameters = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']},
 ]
scores_list = ['precision', 'recall']
for score in scores_list:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X1_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X1_test)
    print(classification_report(y_true, y_pred))
    print()
    print("Final accuracy =", accuracy_score(y_test, clf.predict(X1_test)))

c = 1000
clf_final_2 = SVC(kernel='rbf', C=c, gamma=0.001)
clf_final_2.fit(X1_train, y_train)
y_true, y_pred = y_test, clf_final_2.predict(X1_test)
print(classification_report(y_true, y_pred))
print("Final accuracy =", accuracy_score(y_test, clf_final_2.predict(X1_test)))

print("Number of support Vectors =", len(clf_final_2.support_))
print("Number of suppport vectors for each class:", clf_final_2.n_support_ )
alphas = np.absolute(clf_final_2.dual_coef_)
msv = np.count_nonzero(alphas == c)
print("The margin support vectors =", clf_final_2.dual_coef_.shape[1] - msv)
print("The non-margin support vectors =", msv)
