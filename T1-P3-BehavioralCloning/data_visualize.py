from dataset_generator import *
import numpy as np
from sklearn.model_selection import train_test_split


BATHC_SIZE = 128

# Generate Dataset
data_lines = getImagePath()
image_set, measurement_set = collectData(data_lines)
print(np.shape(image_set))
print(np.shape(measurement_set))


import matplotlib.pyplot as plt
meas = []

plt.hist(measurement_set,np.arange(-1.0,1.05,0.05))
plt.show()

i = np.random.randint(0,1000)
print(measurement_set[i])

path = 'linux_sim/dataset/'
for subfolder in os.listdir(path):

	if os.path.exists(path+subfolder+'/IMG/'+image_set[i]):
		image = cv2.imread(path+subfolder+'/IMG/'+image_set[i])
	else:
		pass

# Normalize
# norm_image = (image/255.0)-0.5
# Crop
# crop_image = norm_image[55:106, 0:319, :]
# Resize
# resize_image = cv2.resize(crop_image,(64,64))

# image = crop_image

cv2.imshow('image',image)
cv2.waitKey(0)

shadow_image, shadow_angle = shadowImage(image,measurement_set[i])
cv2.imshow('shadowImage',shadow_image)
cv2.waitKey(0)

trans_image, trans_angle = translateImage(image,measurement_set[i],80)
print(trans_angle)
cv2.imshow('transImage',trans_image)
cv2.waitKey(0)

flip_image, flip_angle = flipImage(image,measurement_set[i])
print(flip_angle)
cv2.imshow('flipImage',flip_image)
cv2.waitKey(0)

bright_image, bright_angle = changeBrightness(image,measurement_set[i])
print(bright_angle)
cv2.imshow('brightImage',bright_image)
cv2.waitKey(0)

blur_image, blur_angle = blurImage(image,measurement_set[i])
print(blur_angle)
cv2.imshow('blurImage',blur_image)
cv2.waitKey(0)

