import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage
import urllib

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	# NOTE: Be sure to install Pillow with "pip3 install Pillow"
	print('==> Loading image data...')
	img = ndimage.imread(urllib.request.urlopen('http://i.imgur.com/X017qGH.jpg'), flatten=True)

	# --- Shuffling Image ---- 
	imgInput = img.copy()
    imgInput.flatten()
    #adding a random shuffle
	np.random.shuffle(imgInput)

	# reshape the shuffled image
	shuffle_img = imgInput.reshape(img.shape)

	# =============STEP 1: RUNNING SVD ON IMAGES=================
	print('==> Running SVD on images...')

	#-----SVD----
	U, S, V = np.linalg.svd(img)
	U_s, S_s, V_s = np.linalg.svd(shuffle_img)

	# =============STEP 2: SINGULAR VALUE DROPOFF=================
	print('==> Singular value dropoff plot...')
	k = 100
	plt.style.use('ggplot')

	#----- Generate singular value dropoff plot ----
	orig_S_plot = plt.plot(S[0:k], 'b') #blue used 
	shuf_S_plot = plt.plot(S_s[0:k], 'r') #red used

	plt.legend((orig_S_plot, shuf_S_plot), \
		('original', 'shuffled'), loc = 'best')
	plt.title('Singular Value Dropoff for Clown Image')
	plt.ylabel('singular values')
	plt.savefig('dropoff.png', format='png')
	plt.show()

	# =============STEP 3: RECONSTRUCTION=================
	print('==> Reconstruction with different ranks...')
	rank_list = [2, 10, 20]
	plt.subplot(2, 2, 1)
	plt.imshow(img, cmap='Greys_r')
	plt.axis('off')
	plt.title('Original Image')

	for index in range(len(rank_list)):
		k = rank_list[index]
		plt.subplot(2, 2, 2 + index)

		#-----reconstruction images for each rank-----
		recon_img = U[0:-1, 0:k] * np.diag(S)[0:k,0:k] #diagonalizing S
		recon_img = recon_img * * V[0:k,0:-1]
		plt.imshow(recon_img, cmap='Greys_r')

		plt.title('Rank {} Approximation'.format(k))
		plt.axis('off')

	plt.tight_layout()
	plt.savefig('reconstruction.png', format='png')
	plt.show()
