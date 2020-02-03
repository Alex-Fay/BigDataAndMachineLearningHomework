# @author: Alex Fay using code set up provided in HW1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

	# =============part c: Plot data and the optimal linear fit=================
	X = np.array([0, 2, 3, 4])
	y = np.array([1, 3, 6, 8])

	# plot four data points on the plot
	plt.style.use('ggplot')
	plt.plot(X, y, 'ro')

	m_opt = 1.7714
	b_opt = .51428

	# =======generate 100 points along the line of optimal linear fit======
	#	1) Use np.linspace to get the x-coordinate of 100 points
	#	2) Calculate the y-coordinate of those 100 points with the m_opt and
	#	   b_opt, remember y = mx+b
	#	3) Use a.reshape(-1,1), where a is a np.array, to reshape the array
	#	   to appropraite shape for generating plot

	X_space = np.linspace(start =0, stop = 5, num = 100)
	X_space.reshape(-1,1)
	y_space = m_opt * X_space + b_opt #y = mx + b
	y_space.reshape(-1, 1)

	# plot graph and save to folder
	plt.plot(X_space, y_space)
	plt.savefig('hw1pr2c.png', format='png')
	plt.show()
	plt.close()

	# ============= part d: Optimal linear fit with random data points=================
	mu, sigma, sampleSize = 0, 1, 100
	#mu = mean, sigma = shape, size

	# ============ Generate white Gaussian noise =========
	noise = np.random.normal(mu, sigma, sampleSize)
	noise.reshape(-1, 1)
	y_space_rand = (m_opt* X_space) + b_opt + noise

	# calculate the new parameters for optimal linear fit using the points above

	X_similar = np.ones_like((y_space), X_space)	# need to be replaced following hint 1 and 2
	X_space_stacked = np.hstack(X_similar)
	X_linearEqn = X_space_stacked * X_space_stacked

    #np.T stands for matrix transpose
	Y_linearEqn = X_space_stacked.T * y_space_rand
	W_opt = np.linalg.solve(X_linearEqn, Y_linearEqn)

	# calculate predicted values
	y_pred_rand = np.array([(m_rand_opt * x) + b_rand_opt for x in X_space])
	y_pred_rand = y_pred_rand.reshape(-1, 1)

	# get the new m, and new b from W_opt obtained above
	b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)
	plt.plot(X, y, 'ro')
	orig_plot, = plt.plot(X_space, y_space, 'r')

	# plot the generated 100 points with white gaussian noise and the new line
	plt.plot(X_space, y_space_rand, 'bo')
	rand_plot, = plt.plot(X_space, y_pred_rand, 'b')
	plt.legend((orig_plot, rand_plot), \
		('original fit', 'fit with noise'), loc = 'best')
	plt.savefig('hw1pr2d.png', format='png')
	plt.show()
	
