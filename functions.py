import numpy as np
import pandas as pd
import copy

def load_dataset():
    ''' 
    This Function is used to read the h5py file to normal np array formate and decode the file formate
    
    Returns:
    train_set_x_orig -- a pandas DataFrame of training features 
    train_set_y_orig -- a pandas DataFrame of training labels
    test_set_x_orig -- a pandas DataFrame of test features
    test_set_y_orig -- a pandas DataFrame of test labels
    classes -- a numpy array of classes
    '''

    try:
        # Load the train dataset as h5 file
        train_dataset = pd.read_hdf('Datasets/train_catvnoncat.h5')  # Change the file path accordingly
        train_set_x_orig = train_dataset["train_set_x"]
        train_set_y_orig = train_dataset["train_set_y"]

        # Load the test dataset as h5 file
        test_dataset = pd.read_hdf('Datasets/test_catvnoncat.h5')  # Change the file path accordingly
        test_set_x_orig = test_dataset["test_set_x"]
        test_set_y_orig = test_dataset["test_set_y"]

        # Extract the list of classes
        classes = np.array(test_dataset["list_classes"])

        # Reshape the dimension of y set
        train_set_y_orig = train_set_y_orig.values.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.values.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def getting_data():
    try:
        # Loading the data (cat/non-cat)
        train_X_org, train_Y, test_X_org, test_Y, classes = load_dataset()

        if train_X_org is None or train_Y is None or test_X_org is None or test_Y is None or classes is None:
            print("Error loading dataset. Check if the HDF5 files exist and the paths are correct.")
            return None

        # Reshaping the image related to dimension
        train_x_flatten = train_X_org.reshape(train_X_org.shape[0], -1).T
        test_x_flatten = test_X_org.reshape(test_X_org.shape[0], -1).T

        # Standardize the image with 255
        train_X = train_x_flatten / 255
        test_X = test_x_flatten / 255

        return train_X, train_Y, test_X, test_Y, classes

    except Exception as e:
        print(f"Error in getting_data: {e}")
        return None


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    s= 1 / (1+ np.exp(-z))
    return s

def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias) of type float
    """
    
    w=np.zeros(shape=(dim,1))
    b=0.0
    return w, b


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    grads -- dictionary containing the gradients of the weights and bias
            (dw -- gradient of the loss with respect to w, thus same shape as w)
            (db -- gradient of the loss with respect to b, thus same shape as b)
    cost -- negative log-likelihood cost for logistic regression
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    A=sigmoid(np.dot(w.T,X) + b)
    cost= (-1/m) * np.sum((Y * np.log(A)) + ((1-Y)*np.log(1-A)))

    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw= (1/m)*np.dot(X, (A-Y).T)
    db= (1/m)*np.sum(A-Y)

    cost = np.squeeze(np.array(cost))
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule
        w = w-(learning_rate * dw)
        b=b-(learning_rate*db)
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- new image from external (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A=sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
    
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to True to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # getting parameters
    w, b=initialize_with_zeros(X_train.shape[0])
    params, grads, costs=optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w= params['w']
    b=params['b']
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w,b,X_train)

    # Print train/test Errors
    train_accuracy = 100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100
    test_accuracy = 100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100
    # if print_cost:
    #     print("train accuracy: {} %".format(train_accuracy))
    #     print("test accuracy: {} %".format(test_accuracy))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "train_accuracy" : train_accuracy,
         "test_accuracy" : test_accuracy,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
