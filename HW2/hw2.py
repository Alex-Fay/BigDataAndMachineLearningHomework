import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

def linreg(X, y, reg=0.0):
	Identity = np.eye(X.shape[1])
	Identity[0, 0] = 0 # don't regularize the bias term
	W_opt = np.linalg.solve(X.T * X + reg * Identity, X.T * y)
	return W_opt

def predict(W, X):	
	return X * W

def find_RMSE(W, X, y):
	y_pred = predict(W, X)
	diff_y = y - y_pred
	m_dimension = X.shape[0]
	rmse = np.linalg.norm(diff_y, 2) ** 2 / m_dimension
	return np.sqrt(rmse)

def RMSE_vs_lambda(X_train, y_train, X_val, y_val):
	#	This function generates a plot of RMSE vs lambda and returns the
	#	regularization parameter that minimizes RMSE, reg_opt

	RMSE_list = []

	reg_list = np.random.uniform(0, 150, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = lb) for lb in reg_list]
	plt.style.use('ggplot')

	for i in range(len(reg_list)):
		W = W_list[i]
		RMSE_list.append(find_RMSE(W, X_val, y_val))

	# Plot RMSE vs lambda
	RMSE_vs_lambda_plot, = plt.plot(reg_list, RMSE_list)
	plt.setp(RMSE_vs_lambda_plot, color = 'red')
	plt.title('RMSE vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('RMSE')
	plt.savefig('RMSE_vs_lambda.png', format = 'png')
	plt.close()
	print('==> Plotting completed.')

	# Find the regularization value that minimizes RMSE
	optimize_lam = np.min(RMSE_list)
	reg_opt = reg_list[optimize_lam]
	return reg_opt

def norm_vs_lambda(X_train, y_train, X_val, y_val):
	reg_list = []
	W_list = []
	norm_list = []

	reg_list = np.random.uniform(0.0, 150.0, 150)
	reg_list.sort()
	W_list = [linreg(X_train, y_train, reg = lb) for lb in reg_list]

	# Set up plot style
	plt.style.use('ggplot')

	# Plot norm vs lambda
	norm_vs_lambda_plot, = plt.plot(reg_list, norm_list)
	plt.setp(norm_vs_lambda_plot, color='blue')
	plt.title('norm vs lambda')
	plt.xlabel('lambda')
	plt.ylabel('norm')
	plt.show()

# Find the numerical solution in part d
def linreg_no_bias(X, y, reg=0.0):
	t_start = time.time()
	m_dimension = X.shape[0]
	matrix_of_ones = np.eye(m_dimension)

    #calculate b optmized
	weight_opt_step1 = X.T * (np.eye(m_dimension) - np.matrix_of_ones(m_dimension) / m_dimension)
	W_opt = np.linalg.solve(weight_opt_step1 * X + reg * np.eye(weight_opt_step1.shape[0]), weight_opt_step1 * y)
	b_opt = sum((y - X * W_opt)) / m_dimension

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} seconds'.format(\
				t = t_end - t_start))
	return b_opt, W_opt

def grad_descent(X_train, y_train, X_val, y_val, reg=0.0, lr_W=2.5e-12, \
 		lr_b=0.2, max_iter=150, eps=1e-6, print_freq=25):

	m_train, n = X_train.shape
	m_val = X_val.shape[0]

	W = np.zeros((n, 1))
	b = 0
	W_grad = np.ones_like(W)
	b_grad = 1
	obj_train = []
	obj_val = []
	iter_num = 0

	t_start = time.time()

	# start iteration for gradient descent
	while np.linalg.norm(W_grad) > eps and np.linalg.norm(b_grad) > eps and iter_num < max_iter:

		rmseToobj = np.sqrt(rmse_norm ** 2 / m_train)
		obj_train.append(rmseToobj)

		val_rmse = np.sqrt(np.linalg.norm((X_val * W).reshape((-1, 1)) + b - y_val) ** 2 / m_val)
		obj_val.append(val_rmse)

		# gradient
		W_grad = ((X_train.T * X_train + reg * np.eye(n)) * W + X_train.T * (b - y_train)) / m_train
		b_grad = (sum(X_train * W) - sum(y_train) + b * m_train) / m_train

		# update weights and bias
		W -= lr_W * W_grad
		b -= lr_b * b_grad

		# print statements for debugging
		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration{} - training rmse {: 4.4f} - gradient norm {: 4.4E}'.format(\
				iter_num + 1, train_rmse, np.linalg.norm(W_grad)))

		# goes to next iteration
		iter_num += 1

	# Benchmark report
	t_end = time.time()
	print('--Time elapsed for training: {t:4.2f} seconds'.format(\
			t=t_end - t_start))

	# Set up plot style
	plt.style.use('ggplot')

	# generate convergence plot
	train_rmse_plot, = plt.plot(range(iter_num), obj_train)
	plt.setp(train_rmse_plot, color='red')
	val_rmse_plot, = plt.plot(range(iter_num), obj_val)
	plt.setp(val_rmse_plot, color='green')
	plt.legend((train_rmse_plot, val_rmse_plot), \
		('Training RMSE', 'Validation RMSE'), loc='best')
	plt.title('RMSE vs iteration')
	plt.xlabel('iteration')
	plt.ylabel('RMSE')
	plt.savefig('convergence.png', format='png')
	plt.close()
	print('==> Plotting completed.')

	return b, W

###########################################
#	    	Main Driver Function       	  #
###########################################

# You should comment out the sections that
# you have not completed yet

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading data...')

	# Read data
	df = pd.read_csv('https://math189sp19.github.io/data/online_news_popularity.csv', \
		sep=', ', engine='python')

	# split the data frame by type: training, validation, and test
	train_pct = 2.0 / 3
	val_pct = 5.0 / 6

	df['type'] = ''
	df.loc[:int(train_pct * len(df)), 'type'] = 'train'
	df.loc[int(train_pct * len(df)) : int(val_pct * len(df)), 'type'] = 'val'
	df.loc[int(val_pct * len(df)):, 'type'] = 'test'


	# extracting columns into training, validation, and test data
	X_train = np.array(df[df.type == 'train'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_train = np.array(np.log(df[df.type == 'train'].shares)).reshape((-1, 1))

	X_val = np.array(df[df.type == 'val'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_val = np.array(np.log(df[df.type == 'val'].shares)).reshape((-1, 1))

	X_test = np.array(df[df.type == 'test'][[col for col in df.columns \
		if col not in ['url', 'shares', 'type']]])
	y_test = np.array(np.log(df[df.type == 'test'].shares)).reshape((-1, 1))

    #Stack a coloumn of ones to feature data 
	X_train = np.hstack((np.ones(y_train), X_train))
	X_val = np.hstack((np.ones_like(y_val), X_val))
	X_test = np.hstack((np.ones_like(y_test), X_test))

	# Convert data to matrix
	X_train = np.matrix(X_train)
	y_train = np.matrix(y_train)
	X_val = np.matrix(X_val)
	y_val = np.matrix(y_val)
	X_test = np.matrix(X_test)
	y_test = np.matrix(y_test)

	# PART C
	# =============STEP 1: RMSE vs lambda=================
	print('==> Step 1: RMSE vs lambda...')
	# find the optimal regularization parameter
	reg_opt = RMSE_vs_lambda(X_train, y_train, X_val, y_val)
	print('==> The optimal regularization parameter is {reg: 4.4f}.'.format(\
		reg=reg_opt))

	# Find the optimal weights and bias for future use in step 3
	W_with_b_1 = linreg(X_train, y_train, reg=reg_opt)
	b_opt_1 = W_with_b_1[0]
	W_opt_1 = W_with_b_1[1: ]

	# Report the RMSE with the found optimal weights on validation set
	val_RMSE = find_RMSE(W_with_b_1, X_val, y_val)
	print('==> The RMSE on the validation set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(\
		RMSE=val_RMSE))

	# Report the RMSE with the found optimal weights on test set
	test_RMSE = find_RMSE(W_with_b_1, X_test, y_test)
	print('==> The RMSE on the test set with the optimal regularization parameter is {RMSE: 4.4f}.'.format(\
		RMSE=test_RMSE))

	# =============STEP 2: Norm vs lambda=================
	# NOTE: Fill in code in norm_vs_lambda for this step

	print('\n==> Step 2: Norm vs lambda...')
	norm_vs_lambda(X_train, y_train, X_val, y_val)

	# PART D
	# =============STEP 3: Linear regression without bias=================
	# NOTE: Fill in code in linreg_no_bias for this step

	# From here on, we will strip the columns of ones for all data
	X_train = X_train[:, 1:]
	X_val = X_val[:, 1:]
	X_test = X_test[:, 1:]

	# Compare the result with the one from step 1
	# The difference in norm should be a small scalar (i.e, 1e-10)
	print('\n==> Step 3: Linear regression without bias...')
	b_opt_2, W_opt_2 = linreg_no_bias(X_train, y_train, reg=reg_opt)

	# difference in bias
	diff_bias = np.linalg.norm(b_opt_2 - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

	# difference in weights
	diff_W = np.linalg.norm(W_opt_2 -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))

	# PART E
	# =============STEP 4: Gradient descent=================

	print('\n==> Step 4: Gradient descent')
	b_gd, W_gd = grad_descent(X_train, y_train, X_val, y_val, reg=reg_opt)

	# Compare the result from the one from step 1
	# Difference in bias
	diff_bias = np.linalg.norm(b_gd - b_opt_1)
	print('==> Difference in bias is {diff: 4.4E}'.format(diff=diff_bias))

	# difference in weights
	diff_W = np.linalg.norm(W_gd -W_opt_1)
	print('==> Difference in weights is {diff: 4.4E}'.format(diff=diff_W))
