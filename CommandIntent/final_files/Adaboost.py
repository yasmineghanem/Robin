import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:

    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def create_estimator(self):
        # create a decision stump as a weak estimator
        return DecisionTreeClassifier(max_depth=1, random_state=0)

    def fit_and_predict(self, X_train, Y_train, X_test, Y_test):
        # apply AdaBoost on weak estimators
        #Convert range to -1,1
        Y_train = 2 * Y_train - 1
        Y_test = 2 * Y_test - 1
        ## initialize the training and test data with empty array placeholders
        pred_train = np.empty((self.n_estimators, X_train.shape[0]))
        pred_test = np.empty((self.n_estimators, X_test.shape[0]))
        
        # initialize weights
        W = np.ones((X_train.shape[0],)) / X_train.shape[0]

        # loop over the boosting iterations 
        for idx in range(self.n_estimators): 

            # create and fit a new decision stump
            model = self.create_estimator().fit(X_train, Y_train, sample_weight=W)

            # predict classes for the training data and test data
            pred_train_idx = model.predict(X_train)
            pred_test_idx = model.predict(X_test)

            # Calculate the miss Indicator
            miss_indicator = pred_train_idx != Y_train

            # Calculate the error for the current classifier
            cls_err = np.sum(W * miss_indicator) / np.sum(W)

            # Calculate current classifier weight
            cls_alpha = 0.5 * np.log((1 - cls_err) / cls_err)

            # Update the weights 
            W = W * np.exp(-cls_alpha * Y_train * pred_train_idx)

            # Add to the overall predictions
            pred_train[idx] = cls_alpha * pred_train_idx
            pred_test[idx] = cls_alpha * pred_test_idx

            # normalize weights 
            W = W / np.sum(W)

        # Calculate accuracy on train and test sets
        train_accuracy = accuracy_score(Y_train, np.sign(np.sum(pred_train, axis=0)))
        test_accuracy = accuracy_score(Y_test, np.sign(np.sum(pred_test, axis=0)))
        
        return train_accuracy, test_accuracy
    

def get_scores(X_train,y_train, X_test, y_test, n_estimators):
    model = AdaBoost(n_estimators=n_estimators)
    train_accuracy, test_accuracy = model.fit_and_predict(X_train, y_train, X_test, y_test)
    return train_accuracy, test_accuracy