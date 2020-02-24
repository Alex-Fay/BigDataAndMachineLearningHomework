#########################################
#			 Helper Functions	    	#
#########################################

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time

def NLL(X, y, W, reg=0.0):
	eMu = np.exp(X * W)
	probability = eMu / eMu.sum(axis=1).reshape(-1, 1)
	temp = y * np.log(probability)
	NLL = -temp.sum(axis=1).sum()
	NLL += reg * np.diag(W.T * W).sum()
	return NLL

def grad_softmax(X, y, W, reg=0.0):
	#see 2b calculations for more info
	eMu = np.exp(X * W)
	prob = eMu / eMu.sum(axis=1).reshape(-1, 1)
	grad = X.T * (prob - y) + reg * W
	return grad

def predict(X, W):
	eMu = np.exp(X * W)
	probability = eMu / eMu.sum(axis=1).reshape(-1, 1)
	y_pred = np.argmax(probability, axis=1).reshape(-1, 1)
	return y_pred

def get_accuracy(y_pred, y):
	diff = (y_pred == y).astype(int)
	accu = 1. * diff.sum() / len(y)
	return accu

def grad_descent(X, y, reg=0.0, lr=1e-5, eps=1e-6, max_iter=500, print_freq=20):
	# get the shape of the data, and initialize nll_list
	m, n = X.shape
	k = y.shape[1]
	nll_list = []

	W = np.zeros((n, k))
	W_grad = np.ones((n, k))

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	#iteration w/ timer 
while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		#similar to part a, same structure
		nll = NLL(X, y, W, reg=reg)
		if np.isnan(nll): break
		nll_list.append(nll)

		# calculate gradients and update W
		W_gradient = grad_softmax(X, y, W, reg=reg)
		W -= lr * W_gradient
		iter_num += 1

	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} seconds'.format(\
			t=t_end - t_start))

	return W, nll_list

def accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list):
	# initialize the list of accuracy
	accu_list = []

	for reg in lambda_list:
		#using all predefined methods
		W, nll_list = grad_descent(X_train, y_train_OH, reg=reg, lr=2e-5, print_freq=75)
		y_pred = predict(X_test, W)
		acc = get_accuracy(y_pred, y_test)
		accu_list.append(acc)

		print('-- Accuracy is {:2.4f} for lambda = {:2.2f}'.format(accuracy, reg))

	# Plot accuracy vs lambda
	print('==> Printing accuracy vs lambda...')
	plt.style.use('ggplot')
	plt.plot(lambda_list, accu_list)
	plt.title('Accuracy versus Lambda in Softmax Regression')
	plt.xlabel('Lambda')
	plt.ylabel('Accuracy')
	plt.savefig('hw4pr2b_lva.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	opt_lambda_i = np.argmax(accu_list)
	reg_opt = lambda_list[opt_lambda_i]

	return reg_opt

###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':


	# =============STEP 0: LOADING DATA=================
	# NOTE: The data is loaded using the code in p2_data.py. Please make sure
	#		you read the code in that file and understand how it works.

	df_train = data.df_train
	df_test = data.df_test

	X_train = data.X_train
	y_train = data.y_train
	X_test = data.X_test
	y_test = data.y_test

	# stacking an array of ones
	X_train = np.hstack((np.ones_like(y_train), X_train))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# one hot encoder
	enc = OneHotEncoder()
	y_train_OH = enc.fit_transform(y_train.copy()).astype(int).toarray()
	y_test_OH = enc.fit_transform(y_test.copy()).astype(int).toarray()

	# =============STEP 1: Accuracy versus lambda=================

	print('\n\n==> Step 1: Finding optimal regularization parameter...')

	lambda_list = [0.01, 0.1, 0.5, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
	reg_opt = accuracy_vs_lambda(X_train, y_train_OH, X_test, y_test, lambda_list)

	print('\n-- Optimal regularization parameter is {:2.2f}'.format(reg_opt))

	# =============STEP 2: Convergence plot=================

	# run gradient descent to get the nll_list
	W_gd, nll_list_gd = grad_descent(X_train, y_train_OH, reg=reg_opt,\
	 	max_iter=1500, lr=2e-5, print_freq=100)

	print('\n==> Step 2: Plotting convergence plot...')

	# set up style for the plot
	plt.style.use('ggplot')

	# generate the convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	# add legend, title, etc and save the figure
	plt.title('Convergence Plot on Softmax Regression with $\lambda = {:2.2f}$'.format(reg_opt))
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2b_convergence.png', format = 'png')
	plt.show()
	print('==> Plotting completed.')
