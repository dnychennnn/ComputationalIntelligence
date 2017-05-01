from sklearn.metrics import confusion_matrix, mean_squared_error
import random
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc, plot_image
import numpy as np



__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):

	nn_hidden_neuron = 6
	nn = MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(nn_hidden_neuron, ), max_iter=200)
	nn.fit(input2, target2[:,1])
	print(confusion_matrix(target2.T[1], nn.predict(input2)))
	# calculate the weights
	hidden_layer_weights = nn.coefs_[0]
	print(hidden_layer_weights, hidden_layer_weights.shape)
	# print out the mean weight of 6 hidden neurons to observe if there's particular neuron always weigh more.
	for i in range(6):
		print(np.mean(hidden_layer_weights.T[i]))
	plot_hidden_layer_weights(hidden_layer_weights,max_plot=10)
	## TODO
	pass


def ex_2_2(input1, target1, input2, target2):
	## TODO
	nn_hidden_neuron = 20
	train_acc = np.zeros(10)
	test_acc = np.zeros(10)
	nns = []
	for i in range(10):
		random_seed = random.seed()
		nn = MLPClassifier(activation='tanh', solver='adam', hidden_layer_sizes=(nn_hidden_neuron,), max_iter=1000, random_state=random_seed)
		nns.append(nn)
		nn.fit(input1, target1.T[0])
		train_acc[i] = nn.score(input1, target1.T[0])
		test_acc[i] = nn.score(input2, target2.T[0])
		
	plot_histogram_of_acc(train_acc, test_acc)
	print('Best network: ', np.argmax(test_acc))
	# using the best network to calculate the confusion matrix
	conmat = confusion_matrix(target2.T[0], nns[np.argmax(test_acc)].predict(input2))
	print(conmat) 	
	for i in range(len(target2.T[0])):
		if target2.T[0][i] != nns[np.argmax(test_acc)].predict(input2)[i]:
			plot_image(input2[i])
	
	pass

