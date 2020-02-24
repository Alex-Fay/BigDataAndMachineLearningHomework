#@author Alex Fay
#########################################
#			 Helper Functions	    	#
#########################################

import p2_data as data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(x):
#applies logistic function on each entry of input 
	return 1. / (1. + np.exp(-x))

def grad_logreg(X, y, W, reg=0.0):
	logRegEqn = X.T * (sigmoid(X * W) - y) + (reg * W)
	return logRegEqn

def NLL(X, y, W, reg=0.0):
	tempSumNLL = np.multiply(y, np.log( sigmoid(X * W))) + np.multiply((1. - y), np.log(1. - ( sigmoid(X * W))))
	nll = -sum(tempSumNLL) + reg / 2 * np.linalg.norm(W) ** 2
	return nll.item(0)

def grad_descent(X, y, reg=0.0, lr=1e-4, eps=1e-6, max_iter=500, print_freq=20):
	# get the shape of the data, and initiate nll list
	m, n = X.shape
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	W_grad = np.ones_like(W)-

	print('\n==> Running gradient descent...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	while iter_num < max_iter and np.linalg.norm(W_grad) > eps:
		nll = NLL(X, y, W, reg = reg)

		if (nll == 'null' or []): break
		nll_list.append(nll)
		GradientWeight = grad_logreg(X, y, W, reg = reg) #grad_logreg function above
		W -= lr * GradientWeight   #use lr from gradient descent 

		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1

	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running gradient descent: {t:2.2f} seconds'.format(\
			t = t_end - t_start))

	return W, nll_list

def newton_step(X, y, W, reg=0.0):
	grad_result = grad_logreg(X, y, W, reg = reg)
	temp = (np.ndarray((sigmoid(X * W)) * (1. - sigmoid(X * W))))
	diag_matrix = np.diag(np.squeeze(temp)))
	finalMatrix = X.T * diag diag_matrix * X + reg * np.eye(X.shape[1])
	d = np.linalg.solve(finalMatrix, grad_result)
	return d

def newton_method(X, y, reg=0.0, eps=1e-6, max_iter=20, print_freq=5):
	# get the shape of the data, and initiate nll list
	m, n = X.shape
	nll_list = []

	# initialize the weight and its gradient
	W = np.zeros((n, 1))
	step = np.ones_like(W)

	print('==> Running Newton\'s method...')

	# Start iteration for gradient descent
	iter_num = 0
	t_start = time.time()

	# TODO: run gradient descent algorithms

	# HINT: Run the gradient descent algorithm followed steps below
	#	1) Calculate the negative log likelihood at each iteration and
	#	   append the value to nll_list
	#	2) Calculate the gradient for W using newton_step defined above
	#	3) Upgrade W
	#	4) Keep iterating while the number of iterations is less than the
	#	   maximum and the gradient is larger than the threshold

	while iter_num < max_iter and np.linalg.norm(step) > eps:
		nll = NLL(X, y, W, reg = reg)    #get nll
		if (nll == 'null' or []): break
		nll_list.append(nll)
		#find gradients
		weight_step = newton_step(X, y, W, reg = reg)
		W -= weight_step

		# Print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - negative log likelihood {: 4.4f}'.format(\
					iter_num + 1, nll))

		# Goes to the next iteration
		iter_num += 1

	# benchmark
	t_end = time.time()
	print('-- Time elapsed for running Newton\'s method: {t:2.2f} seconds'.format(\
			t = t_end - t_start))

	return W, nll_list

#########################
#		 Step 3			#
#########################

def predict(X, W):
	mu = sigmoid(X @ W)
	return (mu >= 0.5).astype(int)

def get_description(X, y, W):
	y_pred = predict(X, W)
	accuracyCount = 0
	predictCount = 0
	recallCount = 0
	totalPredict = 0
	totalRecall = 0

	for i in range(m):
		result = y.item(i)
		predct = y_pred.item(i)

		if (result == predct):
			accuracyCount += 1

		if result == 1:
			totalRecall += 1

			if predictCount == 1:
				recallCount += 1

		if predct == 1:
			totalPredict += 1

			if result == 1:
				predictCount += 1

#total calculations
	accuracy = 1. * accuracyCount/ m
	precision = 1. * predictCount / totalPredict
	recall = 1. * recallCount / totalRecall
	f1 = 2. * precision * recall / (precision + recall) #F-1 = 2*p*r / (p + r)
	return accuracy, precision, recall, f1

def plot_description(X_train, y_train, X_test, y_test):
	reg_list, acc_list, pred_list, recall_list, f1_list = [], [], [], [], []
	reg_list = [0, 1, 4, 8, 2.5, 9, 3.8, 10.0, 25.0, 30.0, 15.0, 18.7, 2.0, 7.0]
	reg_list.sort()

	#gradient descent
	for i in range(len(reg_list)):
		reg = reg_list[i]
		W_opt, obj = grad_descent(X_train, y_train, reg = reg, lr = 3e-4, print_freq = 100)
		accuracy, precision, recall, f1 = get_description(X_test, y_test, W_opt)

		acc_list.append(accuracy)
		pred_list.append(precision)
		recall_list.append(recall)
		f1_list.append(f1)

	# Generate plots
	# plot accurary versus lambda
	a_vs_lambda_plot, = plt.plot(reg_list, a_list)
	plt.setp(a_vs_lambda_plot, color = 'red')

	# plot precision versus lambda
	p_vs_lambda_plot, = plt.plot(reg_list, p_list)
	plt.setp(p_vs_lambda_plot, color = 'green')

	# plot recall versus lambda
	r_vs_lambda_plot, = plt.plot(reg_list, r_list)
	plt.setp(r_vs_lambda_plot, color = 'blue')

	# plot f1 score versus lambda
	f1_vs_lambda_plot, = plt.plot(reg_list, f1_list)
	plt.setp(f1_vs_lambda_plot, color = 'yellow')

	# Set up the legend, titles, etc. for the plots
	plt.legend((a_vs_lambda_plot, p_vs_lambda_plot, r_vs_lambda_plot, \
		f1_vs_lambda_plot), ('accuracy', 'precision', 'recall', 'F-1'),\
		 loc = 'best')
	plt.title('Testing descriptions')
	plt.xlabel('regularization parameter')
	plt.ylabel('Metric')
	plt.savefig('hw4pr2a_description.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')
	reg_ind = np.max(a_list)
	reg_opt = reg_list[reg_ind]
	return reg_opt

###########################################
#	    	Main Driver Function       	  #
###########################################

if __name__ == '__main__':


	# =============STEP 0: LOADING DATA=================
	# NOTE: The data is loaded using the code in p2_data.py. Please make sure
	#		you read the code in that file and understand how it works.

	# data frame
	df_train = data.df_train
	df_test = data.df_test

	# training data
	X_train = data.X_train
	y_train = data.y_train

	# test data
	X_test = data.X_test
	y_test = data.y_test



	# =============STEP 1: Logistic regression=================
	print('\n\n==> Step 1: Running logistic regression...')

	# splitting data for logistic regression
	# NOTE: for logistic regression, we only want images with label 0 or 1.
	df_train_logreg = df_train[df_train.label <= 1]
	df_test_logreg = df_test[df_test.label <= 1]

	# training data for logistic regression
	X_train_logreg = np.array(df_train_logreg[:][[col for \
		col in df_train_logreg.columns if col != 'label']]) / 256.
	y_train_logreg = np.array(df_train_logreg[:][['label']])

	# testing data for logistic regression
	X_test_logreg = np.array(df_test_logreg[:][[col for \
		col in df_test_logreg.columns if col != 'label']]) / 256.
	y_test_logreg = np.array(df_test_logreg[:][['label']])

	# stacking a column of 1's to both training and testing data
	X_train_logreg = np.hstack((np.ones_like(y_train_logreg), X_train_logreg))
	X_test_logreg = np.hstack((np.ones_like(y_test_logreg), X_test_logreg))


	# ========STEP 1a: Gradient descent=========
	# NOTE: Fill in the code in grad_logreg, NLL and grad_descent for this step

	print('\n==> Step 1a: Running gradient descent...')
	W_gd, nll_list_gd = grad_descent(X_train_logreg, y_train_logreg, reg = 1e-6)


	# ========STEP 1b: Newton's method==========
	# NOTE: Fill in the code in newton_step and newton_method for this step

	print('\n==> Step 1b: Running Newton\'s method...')
	W_newton, nll_list_newton = newton_method(X_train_logreg, y_train_logreg, \
		reg = 1e-6)



	# =============STEP 2: Generate convergence plot=================
	# NOTE: You DO NOT need to fill in any additional helper function for this
	# 		step to run. This step uses what you implemented for the previous
	#		two steps to plot.
	print('\n==> Step 2: Generate Convergence Plot...')
	print('==> Plotting convergence plot...')

	# set up the style for the plot
	plt.style.use('ggplot')

	# plot gradient descent and newton's method convergence plot
	nll_gd_plot, = plt.plot(range(len(nll_list_gd)), nll_list_gd)
	plt.setp(nll_gd_plot, color = 'red')

	nll_newton_plot, = plt.plot(range(len(nll_list_newton)), nll_list_newton)
	plt.setp(nll_newton_plot, color = 'green')

	# add legend, titles, etc. for the plots
	plt.legend((nll_gd_plot, nll_newton_plot), \
		('Gradient descent', 'Newton\'s method'), loc = 'best')
	plt.title('Convergence Plot on Binary MNIST Classification')
	plt.xlabel('Iteration')
	plt.ylabel('NLL')
	plt.savefig('hw4pr2a_convergence.png', format = 'png')
	plt.close()

	print('==> Plotting completed.')



	# =============STEP 3: Generate accuracy/precision plot=================
	# NOTE: Fill in the code in get_description and plot_description for this Step

	print('\nStep 3: ==> Generating plots for accuracy, precision, recall, and F-1 score...')

	# Plot the graph and obtain the optimal regularization parameter
	reg_opt = plot_description(X_train_logreg, y_train_logreg, \
		X_test_logreg, y_test_logreg)

	print('\n==> Optimal regularization parameter is {:4.4f}'.format(reg_opt))
