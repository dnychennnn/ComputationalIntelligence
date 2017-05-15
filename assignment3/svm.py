import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

from sklearn.multiclass import OneVsRestClassifier

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    clf = svm.SVC(kernel='linear')
    clf.fit(x, y)
    plot_svm_decision_boundary(clf, x, y)
    ###########
    pass


def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    xx = np.vstack((x, [4, 0]))
    yy = np.hstack((y, 1))
    clf = svm.SVC(kernel='linear')
    clf.fit(xx, yy)
    plot_svm_decision_boundary(clf, xx, yy)
    pass


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]
    xx = np.vstack((x, [4, 0]))
    yy = np.hstack((y, 1))
    for c in Cs:    
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(xx, yy)
        plot_svm_decision_boundary(clf, xx, yy)


def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    clf = svm.SVC(kernel='linear')    
    clf.fit(x_train, y_train)
    print('score', clf.score(x_test, y_test))
    plot_svm_decision_boundary(clf, x_train, y_train, x_test, y_test)
    pass


def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    degrees = range(1, 21)
    train_scores = np.zeros(20)
    test_scores = np.zeros(20)
    clf = []
    # loop over the polynomial degrees 
    for deg in degrees:
        clf.append(svm.SVC(kernel='poly', degree=deg, coef0=1))
        clf[deg-1].fit(x_train, y_train)
        train_scores[deg-1] = clf[deg-1].score(x_train, y_train)
        test_scores[deg-1] = clf[deg-1].score(x_test, y_test)
    print(train_scores.shape, test_scores.shape, len(clf))
    plot_score_vs_degree(train_scores, test_scores, degrees)
    print('best score: ', np.amax(test_scores), np.argmax(test_scores))
    plot_svm_decision_boundary(clf[np.argmax(test_scores)], x_train, y_train, x_test, y_test)



def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)
    print(gammas.shape)
    train_scores = np.zeros(100)
    test_scores = np.zeros(100)
    clf = []
    # loop over the gammas to set different r value
    for idx, g in enumerate(gammas):
        clf.append(svm.SVC(kernel='rbf', gamma=g))
        clf[idx].fit(x_train, y_train)
        train_scores[idx] = clf[idx].score(x_train, y_train)
        test_scores[idx] = clf[idx].score(x_test, y_test)
    print(train_scores.shape, test_scores.shape, len(clf))
    plot_score_vs_gamma(train_scores, test_scores, gammas)
    print('best score: ', np.amax(test_scores), gammas[np.argmax(test_scores)])
    plot_svm_decision_boundary(clf[np.argmax(test_scores)], x_train, y_train, x_test, y_test)


def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**-3
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########

    # Train the multi-classes linear model using one versus rest
    clf = svm.SVC(kernel='linear', C=3e-4, decision_function_shape='ovr')
    clf.fit(x_train , y_train)
    lin_scores = clf.score(x_test , y_test)
    print(lin_scores)
    # Train the multi-classes rbf models using one versus rest
    gammas = np.arange(1e-5, 1e-3, 0.000099)
    train_scores = np.zeros(10)
    test_scores = np.zeros(10)
    for idx, g in enumerate(gammas):
        clf = svm.SVC(kernel='rbf', C=3e-4, decision_function_shape='ovr', gamma=g)
        clf.fit(x_train, y_train)
        train_scores[idx] = clf.score(x_train, y_train)
        test_scores[idx] = clf.score(x_test, y_test)
    print(train_scores, test_scores)
    plot_score_vs_gamma(train_scores, test_scores, gammas, lin_score_train=lin_scores, baseline=.3)


def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########
    clf = svm.SVC(kernel='linear', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    conmat = confusion_matrix(y_test, clf.predict(x_test))
    print(conmat)
    labels = range(1, 6)
    plot_confusion_matrix(conmat, labels)
    error_rate = np.zeros(5)
    sel_err = np.array([0])  # Numpy indices to select images that are misclassified.
    i = 0  # should be the label number corresponding the largest classification error
    y_pred = clf.predict(x_test)   
    # calculate the error rates
    for label in labels:
        error_rate[label-1] = (np.sum(conmat[label-1])-conmat[label-1][label-1]) / np.sum(conmat[label-1])
    i = np.argmax(error_rate) 
    print(clf.predict(x_test)) 
    # Find out the misclassified images
    misclassified = np.where(y_test != clf.predict(x_test))[0]
    print(misclassified)
    # Find out the misclassified images from the most errored classess
    x = np.where(labels[i] == clf.predict(x_test[misclassified]))[0]
    print(x)
    sel_err = misclassified[x] 
    print(sel_err)
    # Plot with mnist plot
    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='predicted class')
