#!/usr/bin/python

# example from http://www.johnwittenauer.net/machine-learning-exercises-in-python-part-1/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import fileinput

def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        sigma = np.multiply(error, X) # the sum parts of the formulas

        # sigma[0,:] error
        # sigma[1,:] error * milleage[i]
        theta = theta - ((alpha / len(X)) * np.sum(sigma, axis = 0))
        cost[i] = computeCost(X, y, theta)

    return theta, cost

def dataOfCols(data, i):
    return data[data.columns[i]]

def predict(data, xmin, xmax, theta):
    #normalized data
    normd =  (float(data) - float(xmin)) / (float(xmax) - float(xmin))
    return theta[0, 0] + theta[0, 1] * normd

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

# minmax normalization
Xs = X[:,1]
xmin = Xs.min()
xmax = Xs.max()
X = X.astype(float)
X[:,1] = (Xs.astype(float) - float(xmin)) / (float(xmax) - float(xmin))

# variables for gradient descent
alpha = 0.9
# alpha = 0.00000000015
iters = 1000
theta = np.matrix(np.array([0,0]))

# compute gradient descent
theta_result, cost = gradientDescent(X, y, theta, alpha, iters)
print "Theta: ", theta_result
print "Initial cost: ", computeCost(X, y, theta)
print "End cost:     ", computeCost(X, y, theta_result)

# plotting data
firstCol = np.array(X[:,1])
secondCol = dataOfCols(data, 2)
x = np.linspace(firstCol.min(), firstCol.max(), 100)
f = theta_result[0, 0] + (theta_result[0, 1] * x)

fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(x, f, 'r', label = 'Prediction')
ax.scatter(firstCol, secondCol, label = 'Traning Data')
ax.legend(loc = 2)
ax.set_xlabel('Milleage')
ax.set_ylabel('Price')
ax.set_title('Predicted price vs. milleage')
plt.show()

while True:
    line = input("prediction > ")
    if type(line) == int or line.isdigit():
        prediction = predict(int(line), xmin, xmax, theta_result)
        print "The prediction for ", line, " is ", prediction
    else:
        print "Error: please enter an int."
