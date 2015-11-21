# example from http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        blop = np.multiply(error, X[:, 1])
        blop = np.prod(blop, axis = 1)
        b = np.concatenate((error, blop), axis = 1)
        theta = theta - ((alpha / len(X)) * np.sum(b, axis = 0))
        cost[i] = computeCost(X, y, theta)

    return theta, cost

def dataOfCols(data, i):
    return data[data.columns[i]]

# set path
path = os.getcwd() + '/'
if len(sys.argv) > 1:
    path += sys.argv[1]
else:
    path += 'example.csv'

# get data
data = pd.read_csv(path)

data.insert(0, 'Ones', 1)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)

# variables for gradient descent
alpha = 0.01
iters = 100
theta = np.matrix(np.array([0,0]))

# compute gradient descent
theta_result, cost = gradientDescent(X, y, theta, alpha, iters)
print "Theta: ", theta_result
print "Initial cost: ", computeCost(X, y, theta)
print "End cost:     ", computeCost(X, y, theta_result)

# plotting data
firstCol = dataOfCols(data, 1)
secondCol = dataOfCols(data, 2)
x = np.linspace(firstCol.min(), firstCol.max(), 100)
f = theta_result[0, 0] + (theta_result[0, 1] * x)

fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(x, f, 'r', label = 'Prediction')
ax.scatter(firstCol, secondCol, label = 'Traning Data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
