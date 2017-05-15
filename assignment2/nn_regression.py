import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt
import random

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha,plot_bars_early_stopping_mse_comparison

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    mse = mean_squared_error(y, nn.predict(x))
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    # set the hidden neurons to 40
    n_hidden_neurons = 40
    # initiate the regressor
    nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neurons, ), alpha=0, max_iter=200)
    nn.fit(x_train, y_train)
    plot_learned_function(n_hidden_neurons, x_train, y_train, nn.predict(x_train), x_test, y_test, nn.predict(x_test))

    ## TODO
    pass

def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_hidden_neurons = 200
    max_iterations = 200
    train_mses = np.zeros(10)
    # iterate over 10 seeds
    for i in range(10):
        # set the random seed
        random_seed = i
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neurons,), alpha=0, max_iter=max_iterations, random_state=random_seed)
        nn.fit(x_train, y_train)
        # calculate the mses of train and test
        train_mse = calculate_mse(nn, x_train, y_train)
        test_mse = calculate_mse(nn, x_test, y_test)
        train_mses[i] = train_mse
        print('training mse: ', train_mse, '\t', 'testing mse: ', test_mse)
    print('mean of train_mses: ', np.mean(train_mses), '\n', 'max: ', np.amax(train_mses), '\n', 'min: ', np.amin(train_mses), '\n', 'std: ', np.std(train_mses))
    ## TODO
    pass


def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
 
    n_hidden_neurons = [1,2,3,4,6,8,12,20,40]
    max_iterations = 200
    train_mses = np.zeros(shape=(len(n_hidden_neurons), 10))
    test_mses = np.zeros(shape=(len(n_hidden_neurons), 10))

    for idx, n_hidden_neuron in enumerate(n_hidden_neurons):
        for i in range(10):
            random_seed = i
            nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neuron,), alpha=0, max_iter=max_iterations, random_state=random_seed)
            nn.fit(x_train, y_train)
            # calculate the train and test mses
            train_mse = calculate_mse(nn, x_train, y_train)
            test_mse = calculate_mse(nn, x_test, y_test)
            train_mses[idx][i] = train_mse
            test_mses[idx][i] = test_mse
            print('random seed: ', random_seed, 'training mse: ', train_mse, '\t', 'testing mse: ', test_mse)
    for i in range(10):
        print(np.argmin(train_mses.T[i]))
    plot_mse_vs_neurons(train_mses, test_mses, n_hidden_neurons)


    ## TODO
    pass

def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    n_hidden_neurons = [2, 8, 20]
    train_mses = np.zeros(shape=(len(n_hidden_neurons), 1000))
    test_mses = np.zeros(shape=(len(n_hidden_neurons), 1000))
    

    for idx, n_hidden_neuron in enumerate(n_hidden_neurons):
        # set the warm_start to True and max_iter to 1
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neuron, ), alpha=0, max_iter=1, warm_start=True)  
        for i in range(1000):
            nn.fit(x_train, y_train)
            train_mse = calculate_mse(nn, x_train, y_train)
            test_mse = calculate_mse(nn, x_test, y_test)
            train_mses[idx][i] = train_mse
            test_mses[idx][i] = test_mse
            # print('training mse: ', train_mse, '\t', 'testing mse: ', test_mse)

    plot_mse_vs_iterations(train_mses, test_mses, 1000, n_hidden_neurons)

    ## TODO
    pass




def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    n_hidden_neuron = 40
    max_iterations = 200
    alphas = [10**(-8), 10**(-7), 10**(-6), 10**(-5), 10**(-4), 10**(-3), 10**(-2), 10**(-1), 1, 10 , 100]
    train_mses = np.zeros(shape=(len(alphas), 1000))
    test_mses = np.zeros(shape=(len(alphas), 1000))
    print('Start Training ...')
    # iterate over alphas
    for idx, alpha in enumerate(alphas):
        for i in range(10):
            random_seed = i
            nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neuron, ), alpha=alpha, max_iter=max_iterations, random_state=random_seed)
            nn.fit(x_train, y_train)
            train_mse = calculate_mse(nn, x_train, y_train)
            test_mse = calculate_mse(nn, x_test, y_test)
            train_mses[idx][i] = train_mse
            test_mses[idx][i] = test_mse
    print('Plotting ...')
    plot_mse_vs_alpha(train_mses, test_mses, alphas)


    ## TODO
    pass


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    Remember to set alpha and momentum to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The Testingng targets
    :return:
    """
    ## TODO
    # permute the order of the training set
    perm = np.random.permutation(y_train.size)
    x_train = x_train[perm]
    y_train = y_train[perm]
    new_x_train = x_train[:30]
    new_y_train = y_train[:30]
    # split out the validation set
    x_validation = x_train[31:60]
    y_vaildation = y_train[31:60]
    n_hidden_neuron = 40
    alpha = 0.001
    iterations = 100
    # init three array that we need to observe
    test_mse_end = []
    test_mse_early_stopping = []
    test_mse_ideal = []
    for i in range(10):
        test_mses = np.zeros(iterations)
        validation_mses = []
        random_seed = i
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neuron, ), alpha=alpha, max_iter=20, random_state=random_seed, warm_start=True)
        for j in range(iterations):    
            nn.fit(new_x_train, new_y_train) 
            validation_mse = calculate_mse(nn, x_validation, y_vaildation)
            # print('error: ', validation_mse)
            validation_mses.append(validation_mse)
            test_mses[j] = calculate_mse(nn, x_test, y_test)
        test_mse_end.append(test_mses[iterations-1])
        test_mse_early_stopping.append(test_mses[validation_mses.index(min(validation_mses))])
        test_mse_ideal.append(np.amin(test_mses))
        print(test_mse_end[i], test_mse_early_stopping[i], validation_mses.index(min(validation_mses)), test_mse_ideal[i])
    plot_bars_early_stopping_mse_comparison(test_mse_end,test_mse_early_stopping,test_mse_ideal)


        



    pass

def ex_1_2_c(x_train, x_test, y_train, y_test):
    '''
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    '''
    ## TODO
    # permute the order of the traning set
    perm = np.random.permutation(y_train.size)
    x_train = x_train[perm]
    y_train = y_train[perm]
    # split out the training set
    new_x_train = x_train[:30]
    new_y_train = y_train[:30]
    # split out the validation set
    x_validation = x_train[31:60]
    y_vaildation = y_train[31:60]
    alpha = 10**(-3)
    n_hidden_neuron = 40
    train_errors = []
    test_errors = []
    validation_errors = []

    for i in range(10):
        random_seed = i
        validation_mses = []  
        test_mses = [] 
        train_mses = []
        nn = MLPRegressor(activation='logistic', solver='lbfgs', hidden_layer_sizes=(n_hidden_neuron,), alpha=alpha, max_iter=20, random_state=random_seed, warm_start=True)   
        for j in range(10):
            nn.fit(new_x_train, new_y_train)
            validation_mses.append(calculate_mse(nn, x_validation, y_vaildation))
            test_mses.append(calculate_mse(nn, x_test, y_test))
            train_mses.append(calculate_mse(nn, x_train, y_train))
        # find out the optimal seed using early stopping to find when is the lowest validation error
        print(validation_mses.index(min(validation_mses)), test_mses[validation_mses.index(min(validation_mses))], train_mses[validation_mses.index(min(validation_mses))], min(validation_mses))
        train_errors.append(train_mses[validation_mses.index(min(validation_mses))])
        validation_errors.append(min(validation_mses))
        test_errors.append(test_mses[validation_mses.index(min(validation_mses))])
    # print the lowest error of train, test and validation and their index
    print('train: ', np.amin(train_errors), np.argmin(train_errors), '\t', 'test: ', np.amin(test_errors), np.argmin(test_errors), '\t', 'validation', np.amin(validation_errors), np.argmin(validation_errors))
    # print out the standard derivation and mean of the errors
    print('train: ', np.std(train_errors), np.mean(train_errors), '\t', 'test: ', np.std(test_errors), np.mean(test_errors), '\t', 'validation', np.std(validation_errors), np.mean(validation_errors))
    pass