#!/usr/bin/env python
import numpy as np

from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    N, n = x.shape
    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    hypo = sig(x.dot(theta))
    cost = np.array(np.zeros(N))
    cost = np.reshape(cost, (-1, 1))
    # calculate the cost
    for i in range(0, N):
        if y[i] == False:
            cost[i] = -np.log(1 - hypo[i])
        else:
            cost[i] = -np.log(hypo[i])
    # mean each cost
    cost = np.mean(cost.T)
    print("error", cost)
    c = cost

    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #

    # calculate the hypothesis of logistic regression
    hypo = sig(x.dot(theta))
    gradient = np.array(np.zeros(n))
    # delta = h(theta) - yi
    delta = hypo - y
    # summation of delta
    sumdata = x.T.dot(delta)
    # mean over sumdata to get gradient
    gradient = sumdata / N
    g = gradient

    # END TODO
    ###########

    return g
