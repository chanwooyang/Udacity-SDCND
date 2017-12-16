from skimage.feature import hog
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features

# Define a function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features

def hog_feature(img,pix_per_cell,cell_per_block,orient,vis=True,feature_vec=False):

	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
							cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
							feature_vector=feature_vec, block_norm="L2-Hys")
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell,pix_per_cell),
							cells_per_block=(cell_per_block,cell_per_block), visualise=vis,
							feature_vector=feature_vec, block_norm="L2-Hys")
	return features


def extract_features(img, cspace='RGB', pix_per_cell=8,cell_per_block=2,orient=9,
					hog_channel='All',spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256),
					spatial_feat=True, hist_feat=True, hog_feat=True):

	# Color Space Dictionary
	cspace_dict = {'HSV':cv2.COLOR_RGB2HSV, 'LUV':cv2.COLOR_RGB2LUV, 'HLS':cv2.COLOR_RGB2HLS,
					'YUV':cv2.COLOR_RGB2YUV, 'YCrCb':cv2.COLOR_RGB2YCrCb}

	# Create a list to append feature vectors to
	features = []

	# apply color conversion if other than 'RGB'
	if cspace != 'RGB':
		feature_image = cv2.cvtColor(img, cspace_dict[cspace])
	else: 
		feature_image = np.copy(img)

	# Color Feature
	# Apply bin_spatial() to get spatial color features
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		features.append(spatial_features)
	# Apply color_hist() also with a color space option now
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
		features.append(hist_features)

	## HOG Feature
	if hog_feat == True:
		if hog_channel == 'All':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(hog_feature(feature_image[:,:,channel],pix_per_cell,
					cell_per_block,orient,vis=False,feature_vec=True))
			hog_features = np.ravel(hog_features)
		else:
			hog_features = hog_feature(feature_image[:,:,hog_channel],pix_per_cell,
				cell_per_block,orient,vis=False,feature_vec=True)

		features.append(hog_features)

	# Return list of feature vectors
	return np.concatenate(features)


