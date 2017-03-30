#!/usr/bin/env python
import numpy as np

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) and Adaptative gradient descent (GDad) (Logistic Regression)

This file contains generic implementation of gradient descent solvers
The functions are:
- TODO gradient_descent: for a given function with its gradient it finds the minimum with gradient descent
- TODO adaptative_gradient_descent: Same with adaptative learning rate
"""


def gradient_descent(f, df, x0, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param x0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    """
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm

    E_list = np.zeros(max_iter)

    for iter in range(max_iter):
        # update the solution using gradient*learning_rate
        x0 = x0 - df(x0) * learning_rate
        print("error", f(x0), "\t", "learn_rate: ", learning_rate)
        E_list[iter] = f(x0)
    print("sol", x0, "\n", "elist", E_list)
    x = x0

    # END TODO
    ###########

    return x, E_list


def adaptative_gradient_descent(f, df, x0, initial_learning_rate, max_iter):
    """
    Find the optimal solution of the function f using an adaptative gradient descent:

    After every update check whether the cost increased or decreased.
        - If the cost increased, reject the update (go back to the
        previous parameter setting) and multiply the learning rate by 0.7.
        - If the cost decreased, accept the
        update and multiply the learning rate by 1.03.

    The iteration count should be increased after every iteration even if the update was rejected.

    :param f: function to minimize
    :param df: gradient of f
    :param x0: initial point
    :param initial_learning_rate: initial learning rate
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (list of errors), l_rate (The learning rate at the final iteration)
    """

    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm
    #
    E_list = np.zeros(max_iter)
    l_rate = initial_learning_rate

    for iter in range(max_iter):
        # update the solution using gradient*learning_rate
        update_cost = f(x0)
        orig_theta = x0
        x0 = x0 - df(x0) * l_rate

        if f(x0) > update_cost:         # updated cost larger than original cost
            # set the theta to the old one
            x0 = orig_theta
            # multiply the learning rate by 0.7
            l_rate = l_rate * 0.7
        else:                           # updated cost larger than original cost
            # multiply the learning rate by 1.03
            l_rate = l_rate * 1.03

        print("error", f(x0), "\t", "learn_rate: ", l_rate)
        E_list[iter] = f(x0)
    print("sol", x0, "\n", "elist", E_list)
    x = x0

    # END TODO
    ###########

    return x, E_list, l_rate
