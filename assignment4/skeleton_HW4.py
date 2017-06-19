#Filename: HW4_skeleton.py
#Author: Florian Kaum
#Edited: 15.5.2017
#Edited: 19.5.2017 -- changed evth to HW4

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import math
import sys
from scipy.stats import multivariate_normal
import scipy.stats

def plotGaussContour(mu,cov,xmin,xmax,ymin,ymax,title):
	npts = 100
	delta = 0.025
	stdev = np.sqrt(cov)  # make sure that stdev is positive definite

	x = np.arange(xmin, xmax, delta)
	y = np.arange(ymin, ymax, delta)
	X, Y = np.meshgrid(x, y)

	#matplotlib.mlab.bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0, mux=0.0, muy=0.0, sigmaxy=0.0) -> use cov directly
	Z = mlab.bivariate_normal(X,Y,stdev[0][0],stdev[1][1],mu[0], mu[1], cov[0][1])
	plt.plot([mu[0]],[mu[1]],'r+') # plot the mean as a single point
	CS = plt.contour(X, Y, Z)
	plt.clabel(CS, inline=1, fontsize=10)
	plt.title(title)
	plt.show()
	return

def ecdf(realizations):
	x = np.sort(realizations)
	Fx = np.linspace(0,1,len(realizations))
	return Fx,x


#START OF CI ASSIGNMENT 4
#-----------------------------------------------------------------------------------------------------------------------

# positions of anchors
p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
NrAnchors = np.size(p_anchor,0)

# true position of agent
p_true = np.array([[2,-4]])

r_true = np.linalg.norm(p_anchor - p_true[0], axis=1)

# true distance to 4 anchors
dp = np.sqrt(np.sum((p_anchor - p_true)**2, axis=1))

# plot anchors and true position
plt.axis([-6, 6, -6, 6])
for i in range(0, NrAnchors):
	plt.plot(p_anchor[i, 0], p_anchor[i, 1], 'bo')
	plt.text(p_anchor[i, 0] + 0.2, p_anchor[i, 1] + 0.2, r'$p_{a,' + str(i) + '}$')
plt.plot(p_true[0, 0], p_true[0, 1], 'r*')
plt.text(p_true[0, 0] + 0.2, p_true[0, 1] + 0.2, r'$p_{true}$')
plt.xlabel("x/m")
plt.ylabel("y/m")
plt.show()

#1.2) maximum likelihood estimation of models---------------------------------------------------------------------------
#1.2.1) finding the exponential anchor----------------------------------------------------------------------------------
#TODO
def find_expo(data):
	# print(data) 
	for i in range(4):
		plt.plot(data[:,i])
		plt.xlabel('x')
		plt.ylabel('y')
		plt.show()


# #1.2.3) estimating the parameters for all scenarios---------------------------------------------------------------------

#scenario 1
data = np.loadtxt('HW4_1.data',skiprows = 0)
NrSamples = np.size(data,0)
#TODO
sigma = np.var(data.T, axis=1) #calculate the variance(sigma^2)
print('SCENE1', 'Variance: ', sigma)


#scenario 2
data = np.loadtxt('HW4_2.data',skiprows = 0)
NrSamples = np.size(data,0)
find_expo(data)
#TODO

estimator_lambda = NrSamples / np.sum((data[:,0]- r_true[0]).T)
estimator_sigma = np.var(data.T[1:4], axis=1)
print('SCENE2', "Lambda: ", estimator_lambda, 'Variance: ', estimator_sigma)


#scenario 3
data = np.loadtxt('HW4_3.data',skiprows = 0)
NrSamples = np.size(data,0)
#TODO
estimator_lambda = NrSamples / np.sum((data - r_true).T, axis=1)

print('SCENE3', "Lambda: ", estimator_lambda)

# #1.3) Least-Squares Estimation of the Position--------------------------------------------------------------------------
# #1.3.2) writing the function LeastSquaresGN()...(not here but in this file)---------------------------------------------
# #TODO		
def cal_distance(panchor, pt):
	pt = np.asarray(pt.T)
	return np.asmatrix(np.sqrt(np.sum((panchor - pt)**2, axis=1))).T

def cal_p2p(p1, p2):
	return np.sqrt(np.sum(np.power((p1-p2), 2)))


def Jacobian(panchor, pt, dp):
	return (panchor-np.asarray(pt.T))/dp

def  LeastSquaresGN(p_anchor, p_start, r, max_iter, tol):

	p_t = p_start
	for i in range(max_iter):
		dp = cal_distance(p_anchor, p_t)
		Jf = Jacobian(p_anchor, p_t, dp)
		Jft = Jf.T
		p_t_1 = p_t - np.dot(np.dot( np.linalg.inv(np.dot(Jft, Jf)), Jft), (r-dp))
		if cal_p2p(p_t_1, p_t) < tol:
			break
		p_t = p_t_1
	return p_t_1


#1.3.3) evaluating the position estimation for all scenarios------------------------------------------------------------

# choose parameters
# tol = 0.01 # tolerance
# maxIter = 5  # maximum number of iterations
# p_start = np.matrix([[np.random.uniform(-5,5)], [np.random.uniform(-5,5)]])
# print('pstart', p_start)
# # store all N estimated positions
# p_estimated = np.zeros((NrSamples, 2))



# for scenario in range(1,5):
# 	if(scenario == 1):
# 		data = np.loadtxt('HW4_1.data', skiprows=0)
# 	elif(scenario == 2):
# 		data = np.loadtxt('HW4_2.data', skiprows=0)
# 	elif(scenario == 3):
# 		data = np.loadtxt('HW4_3.data', skiprows=0)
# 	elif(scenario == 4):                          
#     #scenario 2 without the exponential anchor
# 		data = np.loadtxt('HW4_2.data', skiprows=0, usecols=(1,2,3))
# 		p_anchor =  np.array([[-5,5],[-5,-5],[5,-5]])
# 	NrSamples = np.size(data, 0)

# 	#perform estimation---------------------------------------
# 	# #TODO
# 	for i in range(0, NrSamples):
# 		if scenario == 4:
# 			r = data[i].reshape((3,1))
# 		else:
# 			r = data[i].reshape((4,1))
# 		p_estimated[i] =  LeastSquaresGN(p_anchor, p_start, r, maxIter, tol).T
# 	# calculate error measures and create plots----------------
# 	#TODO
# 	print(p_estimated - p_true[0])
# 	p_error = np.linalg.norm(p_estimated - p_true[0], axis=1)
# 	print(p_error)
# 	print("mean: ", np.mean(p_error))
# 	print("variance: ", np.var(p_error))
# 	mu = np.mean(p_estimated.T, axis=1)
# 	print(mu)
# 	cov = np.cov(p_estimated.T)
# 	xmax = np.amax(p_estimated[:,0])
# 	xmin = np.amin(p_estimated[:,0])
# 	ymax = np.amax(p_estimated[:,1])
# 	ymin = np.amin(p_estimated[:,1])
# 	title = scenario
# 	plt.scatter(p_estimated[:,0], p_estimated[:,1])
# 	plotGaussContour(mu,cov,xmin,xmax,ymin,ymax,title)
# 	plt.close()	
# 	Fx, x = ecdf(p_error)
# 	plt.xlabel('x')
# 	plt.ylabel('epsilon')
# 	plt.plot(x, Fx)
# 	plt.show()


# 1.4) Numerical Maximum-Likelihood Estimation of the Position (scenario 3)----------------------------------------------
# 1.4.1) calculating the joint likelihood for the first measurement------------------------------------------------------
# TODO
p_anchor = np.array([[5,5],[-5,5],[-5,-5],[5,-5]])
data = np.loadtxt('HW4_3.data', skiprows=0)[0]
jl = np.zeros((201, 201))
r = np.zeros((4))
print(data[0])

for xi in range(-100, 101):
	for yi in range(-100, 101):
		x = xi/20.
		y = yi/20.
		# print(x, y)
		r[0] = np.linalg.norm(p_anchor[0,:] - [x, y])
		r[1] = np.linalg.norm(p_anchor[1,:] - [x, y])
		r[2] = np.linalg.norm(p_anchor[2,:] - [x, y])
		r[3] = np.linalg.norm(p_anchor[3,:] - [x, y])
		jl[xi,yi] = 1

		if((data[0] > r[0]) and (data[1] > r[1]) and (data[2] > r[2]) and (data[3] > r[3])):
			jl[xi,yi] = jl[xi,yi] * estimator_lambda[0] * np.exp( -estimator_lambda[0]*(data[0]-r[0]) )
			jl[xi,yi] = jl[xi,yi] * estimator_lambda[1] * np.exp( -estimator_lambda[1]*(data[1]-r[1]) )
			jl[xi,yi] = jl[xi,yi] * estimator_lambda[2] * np.exp( -estimator_lambda[2]*(data[2]-r[2]) )
			jl[xi,yi] = jl[xi,yi] * estimator_lambda[3] * np.exp( -estimator_lambda[3]*(data[3]-r[3]) )
			
		else:
			jl[xi,yi] = 0
fig = plt.figure()
ax = fig.gca(projection='3d')
x = np.arange(-5, 5.05, 0.05)
y = np.arange(-5, 5.05, 0.05)
X,Y= np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, jl, linewidth=0)
plt.xlabel("x")
plt.ylabel("y")
ax.set_zlabel('Joint-Likelihood')
plt.show()


# print('Maximum likelihood:', jl)

#1.4.2) ML-Estimator----------------------------------------------------------------------------------------------------
# jl = np.zeros((201, 201))
# data = np.loadtxt('HW4_3.data', skiprows=0)
# r = np.zeros((4, 1))
# max_likelihood = np.zeros((2000,2))
# for idx, ms in enumerate(data):
# 	for xi in range(-100, 101):
# 		for yi in range(-100, 101):
# 			x = xi/20.
# 			y = yi/20.
# 			# print(x, y)
# 			r[0] = np.linalg.norm(p_anchor[0,:] - [x, y])
# 			r[1] = np.linalg.norm(p_anchor[1,:] - [x, y])
# 			r[2] = np.linalg.norm(p_anchor[2,:] - [x, y])
# 			r[3] = np.linalg.norm(p_anchor[3,:] - [x, y])
			
# 			if(ms[0] >= r[0] and ms[1] >= r[1] and ms[2] >= r[2] and ms[3] >= r[3]):
# 				jl[xi,yi] = 1
# 				jl[xi,yi] = jl[xi,yi] * estimator_lambda[0] * np.exp( estimator_lambda[0]*(r[0]-ms[0]) )
# 				jl[xi,yi] = jl[xi,yi] * estimator_lambda[1] * np.exp( estimator_lambda[1]*(r[1]-ms[1]) )
# 				jl[xi,yi] = jl[xi,yi] * estimator_lambda[2] * np.exp( estimator_lambda[2]*(r[2]-ms[2]) )
# 				jl[xi,yi] = jl[xi,yi] * estimator_lambda[3] * np.exp( estimator_lambda[3]*(r[3]-ms[3]) )
# 				# print(jl[xi,yi])
# 				# if((r[0]-data[0]) > 0.01 and (r[1]-data[1]) > 0.01 and (r[2]-data[2]) > 0.01 and (r[3]-data[3]) > 0.01):
# 				# 	print(xi, yi)
# 			else:
# 				jl[xi,yi] = 0
# 	max_likelihood[idx] = np.argmax(jl) 
# print(max_likelihood)
#perform estimation---------------------------------------
#TODO

#calculate error measures and create plots----------------
#TODO

#1.4.3) Bayesian Estimator----------------------------------------------------------------------------------------------

#perform estimation---------------------------------------
#TODO

#calculate error measures and create plots----------------
#TODO

