from keras.models import Sequential, load_model, Model
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Convolution2D, BatchNormalization, Activation, Dropout, MaxPooling2D
from keras import metrics, initializations, optimizers

from dataset_generator import *
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BATHC_SIZE = 512
# image_shape = (160, 320, 3)

# Generate Dataset
data_lines = getImagePath()

image_set, measurement_set = collectData(data_lines)
image_set, measurement_set = sklearn.utils.shuffle(image_set, measurement_set)

# train_image_path, valid_image_path, train_angle, valid_angle = train_test_split(image_set,measurement_set, test_size = 0.3)
train_image_path = image_set[:int(len(image_set)*0.7)]
train_angle = measurement_set[:int(len(image_set)*0.7)]
valid_image_path = image_set[int(len(image_set)*0.7):]
valid_angle = measurement_set[int(len(image_set)*0.7):]


training_data = datasetGenerator(train_image_path, train_angle, False, BATHC_SIZE)
validation_data = datasetGenerator(valid_image_path, valid_angle, False, BATHC_SIZE)

if os.path.isfile('model.h5'):
	model = load_model('model.h5')
	print('model loaded')
else:
	print('new model created')
	# Build Network
	model = Sequential()

	# # Normalize
	# model.add(Lambda(lambda x: ((x/255.0)-0.5), input_shape=image_shape))

	# model.add(Cropping2D(cropping=((55,25),(0,0)), input_shape=image_shape))

	# Convolutional Layer 1
	model.add(Convolution2D(24,3,3, subsample=(2,2), init='he_normal', input_shape=(64,64,3)))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(MaxPooling2D((2,2), strides=(1,1)))
	# model.add(Dropout(0.7))

	# Convolutional Layer 2
	model.add(Convolution2D(36,3,3, subsample=(2,2), init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(MaxPooling2D((2,2), strides=(1,1)))
	# model.add(Dropout(0.7))

	# Convolutional Layer 3
	model.add(Convolution2D(48,3,3, subsample=(2,2), init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(MaxPooling2D((2,2), strides=(1,1)))
	# model.add(Dropout(0.2))

	# Convolutional Layer 4
	model.add(Convolution2D(64,3,3, subsample=(1,1), init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(MaxPooling2D((2,2), strides=(2,2)))
	# model.add(Dropout(0.5))

	# Convolutional Layer 5
	model.add(Convolution2D(64,3,3, subsample=(1,1), init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(Dropout(0.7))

	# Flat Layer
	model.add(Flatten())

	# Fully Connected Layer 1
	model.add(Dense(100, init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(Dropout(0.2))

	# Fully Connected Layer 2
	model.add(Dense(50, init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(Dropout(0.2))

	# Fully Connected Layer 3
	model.add(Dense(10, init='he_normal'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	# model.add(Dropout(0.5))

	# Output Layer
	model.add(Dense(1))

adam_op = optimizers.Adam(lr=0.0008)
model.compile(loss='mse', optimizer=adam_op)
history_object = model.fit_generator(training_data, samples_per_epoch=((len(train_image_path)//BATHC_SIZE)*BATHC_SIZE),
validation_data=validation_data, nb_val_samples=len(valid_image_path), nb_epoch=5)

model.save('model.h5')

print(history_object.history.keys())

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE Loss')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['traning set', 'validation set'], loc='upper right')
plt.show()