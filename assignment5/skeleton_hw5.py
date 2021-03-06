#!/usr/bin/env python3
#Filename skeleton_HW5.py
#Author: Christian Knoll, Philipp Gabler
#Edited: 01.6.2017
#Edited: 02.6.2017 -- naming conventions, comments, ...

import numpy as np
import numpy.random as rd
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
from math import pi, exp
from scipy.stats import multivariate_normal


## -------------------------------------------------------    
## ---------------- HELPER FUNCTIONS ---------------------
## -------------------------------------------------------

def sample_discrete_pmf(X, PM, N):
    """Draw N samples for the discrete probability mass function PM that is defined over 
    the support X.

  
    X ... Support of RV -- np.array([...])
    PM ... P(X) -- np.array([...])
    N ... number of samples -- scalar
    """

    assert np.isclose(np.sum(PM), 1.0)
    assert all(0.0 <= p <= 1.0 for p in PM)
    
    y = np.zeros(N)
    cumulativePM = np.cumsum(PM) # build CDF based on PMF
    offsetRand = np.random.uniform(0, 1) * (1 / N) # offset to circumvent numerical issues with cumulativePM
    comb = np.arange(offsetRand, 1 + offsetRand, 1 / N) # new axis with N values in the range ]0,1[
    
    j = 0
    for i in range(0, N):
        while comb[i] >= cumulativePM[j]: # map the linear distributed values comb according to the CDF
            j += 1	
        y[i] = X[j]
        
    return rd.permutation(y) # permutation of all samples


def plot_gauss_contour(mu, cov, xmin, xmax, ymin, ymax, title):
    """Show contour plot for bivariate Gaussian with given mu and cov in the range specified.

    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01], [cov_10, cov_11]]
    xmin, xmax, ymin, ymax ... range for plotting
    """
    
    npts = 500
    deltaX = (xmax - xmin) / npts
    deltaY = (ymax - ymin) / npts
    stdev = [0, 0]

    stdev[0] = np.sqrt(cov[0][0])
    stdev[1] = np.sqrt(cov[1][1])
    x = np.arange(xmin, xmax, deltaX)
    y = np.arange(ymin, ymax, deltaY)
    X, Y = np.meshgrid(x, y)

    Z = mlab.bivariate_normal(X, Y, stdev[0], stdev[1], mu[0], mu[1], cov[0][1])
    plt.plot([mu[0]], [mu[1]], 'r+') # plot the mean as a single point
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline = 1, fontsize = 10)
    plt.title(title)
    # plt.show()


def likelihood_bivariate_normal(X, mu, cov):
    """Returns the likelihood of X for bivariate Gaussian specified with mu and cov.

    X  ... vector to be evaluated -- np.array([[x_00, x_01], ..., [x_n0, x_n1]])
    mu ... mean -- [mu1, mu2]
    cov ... covariance matrix -- [[cov_00, cov_01],[cov_10, cov_11]]
    """
    
    dist = multivariate_normal(mu, cov)
    P = dist.pdf(X)
    return P


## -------------------------------------------------------    
## ------------- START OF  ASSIGNMENT 5 ------------------
## -------------------------------------------------------


def EM(X, M, alpha_0, mu_0, Sigma_0, max_iter):
    
    # TODO
    r = np.zeros((X.shape[0], M))   
    alpha = alpha_0
    mu = mu_0
    sigma = Sigma_0  
    L=[]
    for i in range(max_iter):
        # print(alpha, sigma, mu)
        #E-step
        for m in range(M):
            r[:, m] = alpha[m] * likelihood_bivariate_normal(X, mu[m], sigma[m])
        r = (r.T / np.sum(r, axis=1)).T
        #M-step
        # update Alpha
        alpha = np.sum(r, axis=0) / X.shape[0]
        # update MU
        for m in range(M):
            mu[m,0] = np.sum(r[:,m]*X[:,0])
            mu[m,1] = np.sum(r[:,m]*X[:,1])
        mu = (mu.T/np.sum(r,axis=0)).T
        # update sigma(cov)
        TjStds = np.zeros([len(r),M,2,2]);
        for t in range(len(r)):
            for j in range(M):
                TjStds[t,j] = r[t,j] * np.dot((X[t]-mu[j])[:,np.newaxis],(X[t]-mu[j])[np.newaxis,:])
        sigma = np.sum(TjStds,axis=0)/np.sum(r,axis=0)[:,np.newaxis,np.newaxis]

        # log-likelihood 
        lsum = 0
        for m in range(M):
            lsum += np.log(alpha[m] * likelihood_bivariate_normal(X, mu[m], sigma[m]))
        L.append(np.sum(lsum))
        # print(L)
    print(r)
    plt.figure(3)
    plt.title('Soft Classification')
    plt.scatter(X[:,0], X[:,1   ] , c=np.argmax(r, axis=1))
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1])
    for m in range(M):
        plot_gauss_contour(mu[m], sigma[m], np.amin(X[:,0]), np.amax(X[:,0]), np.amin(X[:,1]), np.amax(X[:,1]), "GMM")
    plt.figure(2)
    plt.xlabel('iterations')
    plt.ylabel('log-likelihood')
    plt.title('X')
    plt.plot(L)
    plt.show()
    return alpha, mu, sigma, L

    pass


def k_means(X, M, mu_0, max_iter):
    # TODO
    mu = mu_0
    closest_mean = np.zeros(X.shape[0])
    D = []
    for i in range(max_iter):
        d = 0
        for idx, x in enumerate(X):
            closest_mean[idx] = np.argmin(np.linalg.norm(x-mu, axis=1))
            d += np.linalg.norm(x - closest_mean[idx])
        D.append(d)
        #update means check converge
        if np.linalg.norm((mu - np.array([X[closest_mean==k].mean(axis=0) for k in range(M)])), axis=1).any() <= 0.01:
            print('converge ...')
            break
        else:
            mu = np.array([X[closest_mean==k].mean(axis=0) for k in range(M)])

    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], c=closest_mean)
    plt.scatter(mu[:,0], mu[:,1], c='r')
    plt.figure(2)
    plt.plot(D)
    plt.show()
    return mu, D
    pass


def sample_GMM(alpha, mu, Sigma, N):
    # TODO
    X = np.random.uniform(0.0, size=(N, 2))
    for m in range(alpha.shape[0]):
        P = alpha * likelihood_bivariate_normal(X, mu[m], Sigma[m])
    print(P)

    return sample_discrete_pmf(X, P, N)
    pass


def main():
    # load data
    X = np.loadtxt('data/X.data', skiprows = 0) # unlabeled data
    a = np.loadtxt('data/a.data', skiprows = 0) # label: a
    e = np.loadtxt('data/e.data', skiprows = 0) # label: e
    i = np.loadtxt('data/i.data', skiprows = 0) # label: i
    o = np.loadtxt('data/o.data', skiprows = 0) # label: o
    y = np.loadtxt('data/y.data', skiprows = 0) # label: y

    # 1.) EM algorithm for GMM:
    # TODO	
    X = X # input data
    M = 5
    alpha = np.ones(M) * 1.0 / M  # -> likelihood of the mixture
    mu = X[np.random.randint(0,X.shape[0],M)] # -> mean of the gaussian
    sigma = np.array([[100,0],[0,100]]) * np.std(X) # -> std of the gaussian
    sigma = [sigma for i in range(M)]
    max_iter = 10
    L = 0 # Log-likelihood   
    alpha, mu, sigma, L = EM(X, M, alpha, mu, sigma, max_iter)
    print(alpha, mu, sigma, L)

    # 2.) K-means algorithm:
    # TODO
    X = X
    M = 5
    centroids = X.copy()
    np.random.shuffle(centroids)
    mu_0 = centroids[:M]
    # mu_0 = np.full((M, 2), 500)
    # print(mu_0)
    max_iter = 1000
    mu, D = k_means(X, M, mu_0, max_iter)
    print('Center: ', mu, 'Distance: ', D)

    # 3.) Sampling from GMM
    # TODO
    # M = 5
    # N = 100
    # alpha = np.ones(M) * 1.0 / M  # -> likelihood of the mixture
    # mu = X[np.random.randint(0,N,M)] # -> mean of the gaussian
    # sigma = np.array([[10,0],[0,10]]) * np.std(X) # -> std of the gaussian
    # sigma = [sigma for i in range(M)]    
    
    # sample_GMM(alpha, mu, sigma, N)
        

    pass


def sanity_checks():
    # likelihood_bivariate_normal
    mu =  [0.0, 0.0]
    cov = [[1, 0.2],[0.2, 0.5]]
    x = np.array([[0.9, 1.2], [0.8, 0.8], [0.1, 1.0]])
    P = likelihood_bivariate_normal(x, mu, cov)
    print(P)

    # plot_gauss_contour(mu, cov, -2, 2, -2, 2, 'Gaussian')

    # sample_discrete_pmf
    PM = np.array([0.2, 0.5, 0.2, 0.1])
    N = 1000
    X = np.array([1, 2, 3, 4])
    Y = sample_discrete_pmf(X, PM, N)
    
    print('Nr_1:', np.sum(Y == 1),
          'Nr_2:', np.sum(Y == 2),
          'Nr_3:', np.sum(Y == 3),
          'Nr_4:', np.sum(Y == 4))


if __name__ == '__main__':
    # to make experiments replicable (you can change this, if you like)
    rd.seed(23434345)
    
    sanity_checks()
    main()
    
