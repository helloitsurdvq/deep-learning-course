import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
#from lr_utils import load_dataset
#%matplotlib inline

#Helper functions: sigmoid
def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s
#print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

#Initializing parameters
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1)) #initialize w as a vector of zeros
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# dim = 2
# w, b = initialize_with_zeros(dim)
# print ("w = " + str(w))
# print ("b = " + str(b))

#Forward and Backward propagation for learning parameters
def propagate(w, b, x, y):
    m = x.shape[1]
    # forward 
    a = sigmoid(np.dot(w.T, x) + b)
    cost = np.sum((-np.log(a)) * y + (-np.log(1 - a))* (1-y)) / m
    # backward
    dw = (np.dot(x, (a - y).T)) / m
    db = np.sum(a - y) / m
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost

# w, b, x, y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
# grads, cost = propagate(w, b, x, y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))

# optimizing w and b by running a gradient descent algorithm
def optimize(w, b, x, y, num_iters, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iters):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, x, y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

# params, grads, costs = optimize(w, b, x, y, num_iters= 100, learning_rate = 0.009, print_cost = False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))

def predict(w, b, x):
    m = x.shape[1]
    y_prediction = np.zeros((1,m))
    w = w.reshape(x.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    a = sigmoid(np.dot(w.T, x) + b)
    
    # solution 1: using else if 
    for i in range(a.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if(a[0, i] <= 0.5): y_prediction[0, i] = 0
        elif(a[0, i] > 0.5): y_prediction[0, i] = 1
    
    # solution 2: using vectorised implementation 
    #y_prediction = (a >= 0.5) * 1.0
    
    assert(y_prediction.shape == (1, m))
    return y_prediction

w = np.array([[0.1124579],[0.23106775]])
b = -0.3
x = np.array([[1,-1.1,-3.2],[1.2,2,0.1]])
print ("predictions = " + str(predict(w, b, x)))

#Merge all functions into a model
def model(x_train, y_train, x_test, y_test, num_iters = 2000, learning_rate = 0.5, print_cost = False):
    # initialize parameters with zeros 
    w, b = initialize_with_zeros(x_train.shape[0])
    # Gradient descent
    params, grads, costs = optimize(w, b, x_train, y_train, 
                                num_iters, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    # w = parameters["w"]
    # b = parameters["b"]
    # Predict test/train set examples
    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, y_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    d = {"costs": costs,
         "y_prediction_test": y_prediction_test, 
         "y_prediction_train" : y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iters}
    
    return d

#d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = False)