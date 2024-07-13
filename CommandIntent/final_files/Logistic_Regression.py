import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_accuracy(true, predicted):    
    truePred = np.sum(true == predicted)
    return truePred / len(true)

class CustomLogisticRegression:
    def __init__(self, lr: int, epochs: int, probability_threshold: float = 0.5, random_state = None):
        self.lr = lr # The learning rate
        self.epochs = epochs # The number of training epochs
        self.probability_threshold = probability_threshold # If the output of the sigmoid function is > probability_threshold, the prediction is considered to be positive (True)
                                                           # otherwise, the prediction is considered to be negative (False)
        self.random_state = random_state # The random state will be used set the random seed for the sake of reproducability
    
    def _prepare_input(self, X):
        # Add a new input with value 1 to each example. It will be multipled by the bias
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.concatenate((ones, X), axis=1)
    
    def _prepare_target(self, y):
        # Convert True to +1 and False to -1
        return np.where(y, 1, -1)

    def _initialize(self, num_weights: int, stdev: float = 0.01):
        # Initialize the weights using a normally distributed random variable with a small standard deviation
        self.weights = np.random.randn(num_weights) * stdev

    def _gradient(self, X, y):
        # Compute and return the gradient of the weights with respect to the loss given the X and y arrays
        error = y - sigmoid(np.dot(X, self.weights))
        weightGradient = np.dot(-X.T, error)
        return weightGradient

    def _update(self, X, y):
        # Apply a single iteration on the weights 
        gradient = self._gradient(X, y)
        self.weights -= self.lr * gradient
        pass

    def fit(self, X, y):
        np.random.seed(self.random_state) # First, we set the seed
        X = self._prepare_input(X) # Then we prepare the inputs & target
        y = self._prepare_target(y)
        self._initialize(X.shape[1]) # Initialize the weights randomly
        for _ in range(self.epochs): # Update the weights for a certain number of epochs
            self._update(X, y)
        return self 
    
    def predict(self, X):
        X = self._prepare_input(X)
        return sigmoid(np.dot(X, self.weights)) > self.probability_threshold


# Function to tune the hyper parameters
def validate(X_train,y_train,lr, epochs):
    validation_size = None #TODO: Choose a size for the validation set as a ratio from the training data
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
    # We will fit the model to only a subset of the training data and we will use the rest to evaluate the performance
    our_model = CustomLogisticRegression(lr=lr, epochs=epochs, random_state=0).fit(X_tr, y_tr)
    # Then, we evaluate the peformance using the validation set
    return calculate_accuracy(y_val, our_model.predict(X_val)) 