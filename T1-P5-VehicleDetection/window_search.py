import numpy as np
import cv2
from feature_extraction import *

def draw_box(img,boxes,color=(0,0,255),thick=3):
	# Make copy of an image
	new_img = np.copy(img)
	# Iterate through bounding boxes
	for box in boxes:
		# Draw Rectangle
		cv2.rectangle(new_img,box[0],box[1],color,thick)

	# Return the image with rectangles
	return new_img

def sliding_window(img,x_start_stop=[None,None],y_start_stop=[None,None],
					xy_window=(64,64),xy_overlap=(0.5,0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	# Compute the span of the region to be searched
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Comput number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	#Compute number of windows in x/y
	nx_buffer = np.int(xy_window[0]*xy_overlap[0])
	ny_buffer = np.int(xy_window[1]*xy_overlap[1])
	nx_windows = np.int((xspan - nx_buffer)/nx_pix_per_step)
	ny_windows = np.int((yspan - ny_buffer)/ny_pix_per_step)
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for y_window in range(ny_windows):
		for x_window in range(nx_windows):
			# Calculate window position
			startx = x_window*nx_pix_per_step+x_start_stop[0]
			endx = startx + xy_window[0]
			starty = y_window*ny_pix_per_step+y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx,starty),(endx,endy)))

	# Return list of windows
	return window_list

def search_windows(img,windows,clf,scaler,cspace='RGB', pix_per_cell=8,cell_per_block=2,
					orient=9,hog_channel=0,spatial_size=(32, 32), hist_bins=32,hist_range=(0, 256),
					spatial_feat=False, hist_feat=False, hog_feat=True):
	# Create an empty list to receive detected windows
	on_windows = []
	# Iterate over all windows in the list
	for window in windows:
		# Extract the test window from the original image
		test_img = cv2.resize(img[window[0][1]:window[1][1],window[0][0]:window[1][0]],(64,64))
		# Extract features for that window using extract_features()
		features = extract_features(test_img,cspace,pix_per_cell,cell_per_block,orient,
					hog_channel,spatial_size, hist_bins, hist_range,spatial_feat, hist_feat,
					hog_feat)
		# Scale extracted features to be fed to the classifier
		test_features = scaler.transform(np.array(features).reshape(1,-1))
		# predict using the classifier
		prediction = clf.predict(test_features)
		# If detected, save the window
		if prediction == 1:
			on_windows.append(window)
	# Return detected windows
	return on_windows


def add_heat(heatmap,boxes):
	# Iterate through boxes
	for box in boxes:
		# Add += 1 for all pixels inside each box
		# Assuming each 'box' takes the form ((x1,y1),(x2,y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

		# Return Heatmap
	return heatmap

def apply_threshold(heatmap,threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return threshold map
	return heatmap

def draw_labeled_boxes(img,labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzerox = np.array(nonzero[1])
		nonzeroy = np.array(nonzero[0])
		# Define a bounding box based on min/max x and y
		box = ((np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy)))
		# Draw the box on the image 
		cv2.rectangle(img,box[0],box[1],(0,0,255),6)
	# Return the image
	return img
