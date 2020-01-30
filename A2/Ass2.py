from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt
import timeit
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X = X / 255
digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
def loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1/m) * L_sum
    return L
nx = X_train.shape[0]
nh = 64
learning_rate = 1
w1 = np.random.randn(nh, nx)
b1 = np.zeros((nh, 1))
w2 = np.random.randn(digits, nh)
b2 = np.zeros((digits, 1))
X = X_train
Y = Y_train
epoch=2000
start=timeit.default_timer()
for i in range(epoch):
    z1 = np.matmul(w1,X) + b1
    result1 = sigmoid(z1)
    z2 = np.matmul(w2,result1) + b2
    result2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)
    cost = loss(Y, result2)
    dz2 = result2-Y
    dw2 = (1./m) * np.matmul(dz2, result1.T)
    db2 = (1./m) * np.sum(dz2, axis=1, keepdims=True)
    dresult1 = np.matmul(w2.T, dz2)
    dz1 = dresult1 * sigmoid(z1) * (1 - sigmoid(z1))
    dw1 = (1./m) * np.matmul(dz1, X.T)
    db1 = (1./m) * np.sum(dz1, axis=1, keepdims=True)
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
stop=timeit.default_timer()
print("Final cost:", cost)
print("Time=",start-stop)
z1 = np.matmul(w1, X_test) + b1
result1 = sigmoid(z1)
z2 = np.matmul(w2, result1) + b2
result2 = np.exp(z2) / np.sum(np.exp(z2), axis=0)
predictions = np.argmax(result2, axis=0)
labels = np.argmax(Y_test, axis=0)
print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),mnist.target, test_size=0.25)
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,test_size=0.1)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(trainData, trainLabels)
score = model.score(valData, valLabels)
print("%d neighbours \t accuracy=%.4f%%" % (1, score * 100))
predictions = model.predict(testData)
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))