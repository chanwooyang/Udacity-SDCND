import os
import csv
import cv2
import numpy as np
import sklearn

# Data Augmentation by flipping  images horizontally
def flipImage(image, measurement):
	flipped_image = cv2.flip(image,1)
	flipped_measurement = (measurement*-1.0)
	return flipped_image, flipped_measurement

def blurImage(image, measurement):
	blurred_image = cv2.GaussianBlur(image,(5,5),0)
	blurred_measurement = measurement
	return blurred_image, blurred_measurement

def changeBrightness(image, measurement):
	'''
	Using 'Gamma Correction' to change luminance of image
	https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
	'''
	gamma = np.random.uniform(1.4,1.7)
	# gamma = 0.3
	bright_image = (((np.array(image)/255)**(1/gamma))*255).astype('uint8')
	bright_measurement = measurement
	return bright_image, bright_measurement

def shadowImage(image, measurement):
	top_y = 320*np.random.uniform()
	top_x = 0
	bot_x = 160
	bot_y = 320*np.random.uniform()
	image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	shadow_mask = 0*image_hls[:,:,1]
	X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
	Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]

	shadow_mask[((X_m-top_x)*(bot_y-top_y)-(bot_x-top_x)*(Y_m-top_y)>=0)]=1

	if np.random.randint(2)==1:
		random_bright = 0.5
		cond1 = shadow_mask==1
		cond0 = shadow_mask==0
		if np.random.randint(2)==1:
			image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
		else:
			image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright

	shadow_image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

	shadow_measurement = measurement

	return shadow_image, shadow_measurement

def translateImage(image, measurement, range):
	trans_x = range*np.random.uniform() - range/2
	trans_angle = measurement + trans_x/range*2*0.2

	trans_y = 0

	trans_Matrix = np.float32([[1,0,trans_x],[0,1,trans_y]])
	trans_image = cv2.warpAffine(image, trans_Matrix, (320,160))

	return trans_image, trans_angle


# Get all data
path = 'linux_sim/dataset/'
def getImagePath():
	data_lines = []
	for subfolder in os.listdir(path):
		print(subfolder)
		with open(path+subfolder+'/driving_log.csv') as csvfile:
			reader = csv.reader(csvfile)
			for line in reader:
				data_lines.append(line)
	return data_lines

angle_threshold = 0.05

def collectData(dataLines):
	imagePaths = []
	measurements = []

	keep_prob = 0.4
	correction_factor = 0.25

	for dataLine in dataLines:
		for i in range(3):
			measurement = float(dataLine[3])
			if abs(measurement) > angle_threshold:
				source_path = dataLine[i]
				imagePath = source_path.split('/')[-1]
				imagePaths.append(imagePath)
				if i == 1:
					measurements.append(measurement + correction_factor)
				elif i == 2:
					measurements.append(measurement - correction_factor)
				else:
					measurements.append(measurement)

			elif abs(measurement) <= angle_threshold and np.random.rand() < keep_prob:
				source_path = dataLine[i]
				imagePath = source_path.split('/')[-1]
				imagePaths.append(imagePath)	
				if i == 1:
					measurements.append(measurement + correction_factor)
				elif i == 2:
					measurements.append(measurement - correction_factor)
				else:
					measurements.append(measurement)

	assert(len(imagePaths) == len(measurements))

	return imagePaths, measurements

def trimData(imagePaths, measurements):

	num_bins = 100
	hist, bins = np.histogram(measurements, num_bins)
	keep_prob=[]
	target = np.max(hist)*0.5
	for i in range(num_bins):
		if hist[i] < target:
			keep_prob.append(1.)
		else:
			keep_prob.append(target/hist[i])

	remove_list = []
	for i, angle in enumerate(measurements):
		for j in range(num_bins):
			if angle > bins[j] and angle <= bins[j+1]:
				if np.random.rand() > keep_prob[j]:
					remove_list.append(i)
	imagePaths = np.delete(imagePaths, remove_list)
	measurements = np.delete(measurements, remove_list)

	assert(len(imagePaths) == len(measurements))

	return imagePaths, measurements

def datasetGenerator(imagePaths, measurements, validation_flag, BATCH_SIZE):

	image_paths, angles = sklearn.utils.shuffle(imagePaths, measurements)
	X_train = []
	y_train = []
	while 1:
		for i, imagePath in enumerate(image_paths):
			for subfolder in os.listdir(path):
				if os.path.exists(path+subfolder+'/IMG/'+imagePath):
					raw_image = cv2.imread(path+subfolder+'/IMG/'+imagePath)
				else:
					pass
			image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) # 'cv2.imread' reads images as BGR
			angle = angles[i]

			dice1 = np.random.rand()
			dice2 = np.random.rand()

			if abs(angle) > 0.3 and dice2 < 0.5:
				flip_image, flip_angle = flipImage(image, angle)
				true_image = flip_image
				true_angle = flip_angle
			else:
				true_image = image
				true_angle = angle 
 
			if dice1 < 0.2and len(X_train) < BATCH_SIZE:
				X_train.append(true_image)
				y_train.append(true_angle)

				# print('normal')
				# cv2.imshow('flipImage',true_image)
				# cv2.waitKey(0)

			elif dice1 >=0.2 and dice1 < 0.4 and len(X_train) < BATCH_SIZE:
				blur_image, blur_angle = blurImage(true_image, true_angle)
				X_train.append(blur_image)
				y_train.append(blur_angle)

				# print('blur')
				# cv2.imshow('flipImage',blur_image)
				# cv2.waitKey(0)

			elif dice1 >=0.4 and dice1 < 0.6 and len(X_train) < BATCH_SIZE:
				bright_image, bright_angle = changeBrightness(true_image, true_angle)
				X_train.append(bright_image)
				y_train.append(bright_angle)

				# print('bright')
				# cv2.imshow('flipImage',bright_image)
				# cv2.waitKey(0)

			elif dice1 >=0.6 and dice1 < 0.8 and len(X_train) < BATCH_SIZE:
				shadow_image, shadow_angle = shadowImage(true_image, true_angle)
				X_train.append(shadow_image)
				y_train.append(shadow_angle)				

				# print('shadow')
				# cv2.imshow('flipImage',shadow_image)
				# cv2.waitKey(0)

			elif dice1 >= 0.8 and dice1 < 1.0 and len(X_train) < BATCH_SIZE:
				trans_image, trans_angle = translateImage(true_image, true_angle, 80)
				X_train.append(trans_image)
				y_train.append(trans_angle)				
		
				# print('trans')
				# cv2.imshow('flipImage',trans_image)
				# cv2.waitKey(0)

			if len(X_train) == BATCH_SIZE:
				assert(len(X_train) == len(y_train))
				yield sklearn.utils.shuffle(np.array(X_train),np.array(y_train))
				X_train = []
				y_train = []
