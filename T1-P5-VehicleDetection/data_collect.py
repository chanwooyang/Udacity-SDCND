import numpy as np
import cv2
import os
import glob
import pickle
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from feature_extraction import *

def collect_data():
	dataset_path = 'dataset/'
	if os.path.exists(dataset_path):
		car_images = glob.glob(dataset_path+'vehicles/**/*.png')
		noncar_images = glob.glob(dataset_path+'non-vehicles/**/*.png')

		print('Number of car images: {0}'.format(len(car_images)))
		print('Number of non-car images: {0}'.format(len(noncar_images)))

		
		car_img = cv2.imread(car_images[np.random.randint(0,len(car_images))])
		print('Car image size: {0}'.format(car_img.shape))
		noncar_img = cv2.imread(noncar_images[np.random.randint(0,len(car_images))])
		print('Non-Car image size: {0}'.format(noncar_img.shape))

		return car_images, noncar_images
	else:
		raise TypeError("Dataset directory does not exist")

def dataset_split(car_image_paths,noncar_image_paths,cspace='RGB',pix_per_cell=8,cell_per_block=2,
				orient=9,hog_channel='All',spatial_size=(32, 32),hist_bins=32, 
				hist_range=(0, 256),spatial_feat=True, hist_feat=True, hog_feat=True):

	if os.path.isfile('ProcessedData.p'):
		
		data_file = 'ProcessedData.p'
		with open(data_file, mode='rb') as f:
		    data = pickle.load(f)
		X_train = data['X_train']
		X_test = data['X_test']
		y_train = data['y_train']
		y_test = data['y_test']
		X_scaler = data['X_scaler']
		
		return X_train,X_test,y_train,y_test,X_scaler

	else:
		car_features = []
		noncar_features = []

		print('Extracting car features...')
		t = time.time()
		for car_path in car_image_paths:
			car_img = cv2.imread(car_path)
			car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2RGB)
			car_features.append(extract_features(car_img,cspace,pix_per_cell,cell_per_block,orient,
				hog_channel,spatial_size,hist_bins,hist_range,spatial_feat,hist_feat,hog_feat))
		t1 = time.time()
		print('Car Feature extraction complete. ',round(t1-t,2), ' seconds to extract features')

		print('Extracting noncar features...')
		t2 = time.time()
		for noncar_path in noncar_image_paths:
			noncar_img = cv2.imread(noncar_path)
			noncar_img = cv2.cvtColor(noncar_img,cv2.COLOR_BGR2RGB)
			noncar_features.append(extract_features(noncar_img,cspace,pix_per_cell,cell_per_block,orient,
				hog_channel,spatial_size,hist_bins,hist_range,spatial_feat,hist_feat,hog_feat))
		t3 = time.time()
		print('Noncar Feature extraction complete. ',round(t3-t2,2), ' seconds to extract features')

		X = np.vstack((car_features, noncar_features)).astype(np.float64)
		# Fit a per-column scaler
		X_scaler = StandardScaler().fit(X)
		# Apply the scaler to X
		scaled_X = X_scaler.transform(X)

		y = np.hstack((np.ones(len(car_features)),np.zeros(len(noncar_features))))

		assert(len(scaled_X) == len(y))

		X_train,X_test,y_train,y_test = train_test_split(scaled_X,y,test_size=0.2)


		pickle_file = 'ProcessedData.p'
		print('Saving data to pickle file...')
		try:
			with open(pickle_file, 'wb') as pfile:
				pickle.dump(
					{
						'X_train': X_train,
						'X_test': X_test,
						'y_train': y_train,
						'y_test': y_test,
						'X_scaler': X_scaler
					},
					pfile, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', pickle_file, ':', e)
			raise

	return X_train,X_test,y_train,y_test,X_scaler
