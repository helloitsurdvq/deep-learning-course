import numpy as np
import matplotlib.pyplot as plt
#from symbol import parameters
from testcase import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # set a seed so that the results are consistent

X, Y = load_planar_dataset()
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);
#plt.show()
shape_X = X.shape
shape_Y = Y.shape
m = (X.size)/shape_X[0]  # training set size

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
#plt.show()
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
   '% ' + "(percentage of correctly labelled datapoints)")

#neural network model 
#1. defining the neural network structure
def layer_sizes(x, y):
       n_x = x.shape[0] #size of input layer
       n_h = 4 #size of hidden layer
       n_y = y.shape[0] #size of output layer 
       return (n_x, n_h, n_y)
              
# x_assess, y_assess = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(x_assess, y_assess)
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))

#2. initialize the model's parameters
def initialize_parameters(n_x, n_h, n_y):
       #np.random.seed(2)
       w1 = np.random.randn(n_h,n_x) * 0.01
       b1 = np.zeros((n_h, 1))
       w2 = np.random.randn(n_y,n_h) * 0.01
       b2 = np.zeros((n_y, 1))
       assert (w1.shape == (n_h, n_x))
       assert (b1.shape == (n_h, 1))
       assert (w2.shape == (n_y, n_h))
       assert (b2.shape == (n_y, 1))
       
       parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
       return parameters

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("w1 = " + str(parameters["w1"]))
# print("b1 = " + str(parameters["b1"]))
# print("w2 = " + str(parameters["w2"]))
# print("b2 = " + str(parameters["b2"]))

#3. the loop
def sigmoid(z):
       s = 1/(1+np.exp(-z))
       return s
def forward_propagation(x, parameters):
       w1 = parameters["W1"]
       b1 = parameters["b1"]
       w2 = parameters["W2"]
       b2 = parameters["b2"]
       
       z1 = np.dot(w1, x) + b1
       a1 = np.tanh(z1)
       z2 = np.dot(w2, a1) + b2
       a2 = sigmoid(z2) 
       
       assert(a2.shape == (1, x.shape[1]))
       cache = {"z1": z1, "a1": a1, "z2": z2, "a2": a2}
       return a2, cache

x_assess, parameters = forward_propagation_test_case()
a2, cache = forward_propagation(x_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours. 
# print(np.mean(cache['z1']) ,np.mean(cache['a1']),np.mean(cache['z2']),np.mean(cache['a2']))

#cost function
def compute_cost(a2, y, parameters):
       m= y.shape[1]
       logprobs = np.multiply(np.log(a2), y) + np.multiply(np.log(1 - a2), (1 - y))
       cost = (np.sum(logprobs)) / (-m)
       cost = float(np.squeeze(cost)) #remove redundant dims
       assert(isinstance(cost, float))
       return cost

# a2, y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(a2, y_assess, parameters)))

#backward propagation
def backward_propagation(parameters, cache, x, y):
       m = x.shape[1]

       W1 = parameters["W1"]
       b1 = parameters["b1"]
       W2 = parameters["W2"]
       b2 = parameters["b2"]
       
       A1 = cache["A1"]
       A2 = cache["A2"]
       Z1 = cache["Z1"]
       Z2 = cache["Z2"]
       
       dz2 = A2 - y
       dw2 = (1/m) * np.dot(dz2,A1.T)
       db2 = (1/m) *(np.sum(dz2,axis=1,keepdims=True))
       dz1 = np.dot(W2.T,dz2) * (1 - np.power(A1,2))
       dw1 = (1/m) *(np.dot(dz1,x.T))
       db1 = (1/m) *(np.sum(dz1, axis=1, keepdims=True))
       
       grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
       return grads

# parameters, cache, x_assess, y_assess = backward_propagation_test_case()

# grads = backward_propagation(parameters, cache, x_assess, y_assess)
# print ("dw1 = "+ str(grads["dw1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dw2 = "+ str(grads["dw2"]))
# print ("db2 = "+ str(grads["db2"]))

def update_parameters(parameters, grads, learning_rate):
       w1 = parameters["W1"]
       b1 = parameters["b1"]
       w2 = parameters["W2"]
       b2 = parameters["b2"]

       dw1 = grads["dW1"]
       db1 = grads["db1"]
       dw2 = grads["dW2"]
       db2 = grads["db2"]

       w1 = w1 - learning_rate * dw1
       b1 = b1 - learning_rate * db1
       w2 = w2 - learning_rate * dw2
       b2 = b2 - learning_rate * db2
       
       parameters = {"w1": w1,"b1": b1, "w2": w2, "b2": b2}
       return parameters

# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads,1.2)
# print("w1 = " + str(parameters["w1"]))
# print("b1 = " + str(parameters["b1"]))
# print("w2 = " + str(parameters["w2"]))
# print("b2 = " + str(parameters["b2"]))

#build neural network model
def neuralNetwork_model(x, y, n_h, learning_rate, num_iters = 10000, print_cost = False):
       n_x = layer_sizes(x, y)[0]
       n_y = layer_sizes(x, y)[2]
       
       parameters = initialize_parameters(n_x, n_h, n_y)
       w1 = parameters["W1"]
       b1 = parameters["b1"]
       w2 = parameters["W2"]
       b2 = parameters["b2"]
       
       for i in range(0, num_iters): #gradient descent
              a2, cache = forward_propagation(x, parameters)
              cost = compute_cost(a2, y, parameters)
              grads = backward_propagation(parameters, x, y)
              parameters = update_parameters(parameters, grads, learning_rate)
              if print_cost and i%1000 == 0:
                     print("cost after iteration %i: %f" %(i, cost))
       return parameters
       
def predict(x, parameters):
       a2, cache = forward_propagation(x, parameters)
       predictions = (a2 > 0.5)
       return predictions

# Build a model with a n_h-dimensional hidden layer
parameters = neuralNetwork_model(X, Y, 4, 1.2 , num_iters = 10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(x.T, parameters), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')