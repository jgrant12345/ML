"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2021 Jan 14
Description : Titanic
"""

# Use only the provided packages!
import math

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns

######################################################################
# globals
######################################################################

SOLID = ()
DASHED = (5, 5)

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key = lambda val: val[1]) # val = (val,count)
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # insert your RandomClassifier code
        
        ### ========== TODO : END ========== ###
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # insert your RandomClassifier code
        
        y = None
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################

def error(clf, X, y, ntrials=100, nfolds=10, train_size=1.0) :
    """
    Computes the classifier error averaged over ntrials trials and nfolds per trial.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
        train_size  -- float (between 0.0 and 1.0),
                       the proportion of the dataset to include in the train split
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
                       (technically cross-validation error but sklearn calls it test set)
    """
    
    # initialize train and test scores for each trial and fold
    train_scores = np.empty((ntrials, nfolds))
    test_scores = np.empty((ntrials, nfolds))
    
    ### ========== TODO : START ========== ###
    # part c: compute errors over trials and folds
    # hint: use Stratified KFold (set three parameters)
    # professor's solution: 10 lines
    
    # part f: vary training set size
    # for a given fold, splice train indices to keep start of list, rounding down if needed
    # professor's solution: 1 line
    
    
    ### ========== TODO : END ========== ###
    
    # average over the trials
    train_error = 1 - float(sum(train_scores.flatten())) / len(train_scores.flatten())
    test_error = 1 - float(sum(test_scores.flatten())) / len(test_scores.flatten())
    
    return train_error, test_error


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic_train = pd.read_csv("../data/titanic_train.csv")
    target_name = "Survived"
    X_train = titanic_train.drop([target_name], axis=1).to_numpy()
    y_train = titanic_train[target_name].to_numpy()
    Xnames = titanic_train.columns.drop([target_name]).values
    yname = target_name
    n,d = X_train.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X_train, y_train)      # fit training data using the classifier
    y_pred = clf.predict(X_train)  # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y_train, y_pred, normalize=True)
    print(f'\t-- training error: {train_error:.3f}')
    
    
    
    ### ========== TODO : START ========== ###
    # part a: evaluate training error of Decision Tree classifier
    # use random_state=1234 to ensure consistency with solutions (set two parameters total)
    # professor's solution: 5 lines
    print('Classifying using Decision Tree...')
    
    ### ========== TODO : END ========== ###
    
    
    
    # note: uncomment following lines to output Decision Tree graph
    # be sure to re-comment before submission
    """
    # save the classifier -- requires GraphViz and pydot
    import io
    import pydotplus
    from sklearn import tree
    dot_data = io.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames,
                         class_names=["Died", "Survived"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("titanic_dtree.pdf") 
    """
    
    
    
    #========================================
    # compute average training and test error of classifiers using cross-validation
    print('Investigating various classifiers...')
    
    ### ========== TODO : START ========== ###
    # part d: evaluate using error(...)
    # professor's solution: 6 lines
    
    ### ========== TODO : END ========== ###
    
    
    
    #========================================
    # investigate decision tree classifier with various depths
    print('Investigating depths...')
    
    depths = np.arange(1, 22, 5)
    train_errors_depths = []
    test_errors_depths = []
    for depth in depths :
        ### ========== TODO : START ========== ###
        # part e: set parameters for classifier
        # professor's solution: 1 line
        clf = DecisionTreeClassifier()
        ### ========== TODO : END ========== ###
        train_error, test_error = error(clf, X_train, y_train)
        train_errors_depths.append(train_error)
        test_errors_depths.append(test_error)
    
    # plot the results
    df = pd.DataFrame({'depth': depths,
                       'train': train_errors_depths,
                       'cv': test_errors_depths})
    df = df.melt(id_vars=['depth'], var_name='dataset', value_name='error')
    ax = sns.lineplot(data=df, x='depth', y='error', style='dataset', dashes=[DASHED, SOLID])
    ax.set(xlim=(min(depths), max(depths)))
    ax.set(ylim=(0, 0.5))
    plt.savefig("../plots/depth.pdf")
    #plt.show()   # pause plot
    plt.clf()
    
    ### ========== TODO : START ========== ###
    # part e: determine optimal depth
    # professor's solution: 3 lines
    
    ### ========== TODO : END ========== ###
    
    
    
    #========================================
    # investigate decision tree classifier with various training set sizes
    print('Investigating training set sizes...')
    
    ### ========== TODO : START ========== ###
    # part g: set parameters for classifiers
    # professor's solution: 3 lines
    
    clfs = {'full':        DecisionTreeClassifier(), # full depth
            'single leaf': DecisionTreeClassifier(), # single leaf
            'optimal':     DecisionTreeClassifier()} # optimal (tuned) depth
    ### ========== TODO : END ========== ###
    
    train_sizes = np.linspace(.1, 1.0, 5)
    errors = []
    for train_size in train_sizes :
        for clf_name, clf in clfs.items() :
            train_error, test_error = error(clf, X_train, y_train, train_size=train_size)
            errors.append({'train_size': train_size, 'depth': clf_name,
                           'train': train_error,     'cv': test_error})
    
    # plot the results
    df = pd.DataFrame(errors)
    df = df.melt(id_vars=['train_size', 'depth'], var_name='dataset', value_name='error')
    ax = sns.lineplot(data=df, x='train_size', y='error', style='dataset', hue='depth', dashes=[DASHED, SOLID])
    ax.set(ylim=(0, 0.5))
    plt.savefig("../plots/learningcurve.pdf")
    #plt.show()   # pause plot
    plt.clf()
    
    
    
    ### ========== TODO : START ========== ###
    # part h: contest
    # uncomment pd.DataFrame.to_csv(...) and change the filename
    
    # evaluate on test data
    titanic_test = pd.read_csv("../data/titanic_test.csv")
    X_test = titanic_test
    y_pred = clf.predict(X_test)   # take the trained classifier and run it on the test data
    #pd.DataFrame(y_pred, columns=[target_name]).to_csv("../data/yjw_titanic.csv", index=False)
    
    ### ========== TODO : END ========== ###
    
    
    
    print('Done')

if __name__ == "__main__":
    main()