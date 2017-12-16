import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data_collect import *
from feature_extraction import *
from window_search import *
from classifier import *
from pipeline import *

# car_images, noncar_images = collect_data()

# # Display sample data
# n_sample = 8
# plt.figure(figsize=(12,5))
# for i in range(n_sample):
# 	img = cv2.imread(car_images[np.random.randint(0,len(car_images))])
# 	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 	plt.subplot(241+i,title = 'car')
# 	plt.imshow(img)
# plt.show()

# plt.figure(figsize=(12,5))
# for i in range(n_sample):
# 	img = cv2.imread(noncar_images[np.random.randint(0,len(car_images))])
# 	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 	plt.subplot(241+i,title = 'non-car')
# 	plt.imshow(img)
# plt.show()
# #########################################################################################
# # Display HOG image
# pix_per_cell = 8
# cell_per_block = 2
# orient = 9

# car_img = cv2.imread(car_images[np.random.randint(0,len(car_images))])
# car_img = cv2.cvtColor(car_img,cv2.COLOR_BGR2GRAY)
# noncar_img = cv2.imread(noncar_images[np.random.randint(0,len(noncar_images))])
# noncar_img = cv2.cvtColor(noncar_img,cv2.COLOR_BGR2GRAY)

# _, car_hog = hog_feature(car_img,pix_per_cell,cell_per_block,orient)
# _, noncar_hog = hog_feature(noncar_img,pix_per_cell,cell_per_block,orient)

# plt.figure(figsize=(7,7))
# plt.subplot(221, title='car img')
# plt.imshow(car_img,cmap='gray')
# plt.subplot(222, title='noncar img')
# plt.imshow(noncar_img,cmap='gray')
# plt.subplot(223, title='car hog img')
# plt.imshow(car_hog,cmap='gray')
# plt.subplot(224, title='noncar hog img')
# plt.imshow(noncar_hog,cmap='gray')
# plt.show()
############################################################################################
# Draw All Boxes

# test_img = mpimg.imread('./test_images/test6.jpg')

# span_ratio1,span_ratio2,span_ratio3,span_ratio4 = 1.25,1.25,1,1
# wsize1,wsize2,wsize3,wsize4 = 64, 128, 192, 256
# ystart1,ystart2,ystart3,ystart4 = 400, 390, 380, 370
# overlap1,overlap2,overlap3,overlap4 = 0.75,0.75,0.75,0.75
# x_start_stop=[[None,None],[None,None],[None,None],[None,None]]
# y_start_stop=[[ystart1,ystart1+wsize1*span_ratio1],[ystart2,ystart2+wsize2*span_ratio2],[ystart3,ystart3+wsize3*span_ratio3],[ystart4,ystart4+wsize4*span_ratio4]]
# xy_window=[(wsize1,wsize1),(wsize2,wsize2),(wsize3,wsize3),(wsize4,wsize4)]
# xy_overlap=[(overlap1,overlap1),(overlap2,overlap2),(overlap3,overlap3),(overlap4,overlap4)]

# output = []
# for i in range(len(x_start_stop)):
#     boxes = sliding_window(test_img,x_start_stop[i],y_start_stop[i],xy_window[i],xy_overlap[i])
#     output.append(draw_box(test_img,boxes,color=(np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))))

# for i in range(len(output)):
#     plt.imshow(output[i])
#     plt.show()

# ############################################################################################
# Draw Only Detected Boxes
# cspace='YUV'
# pix_per_cell=16
# cell_per_block=2
# orient=9
# hog_channel='All'
# spatial_size=(32, 32)
# hist_bins=32
# hist_range=(0, 256)
# spatial_feat=True
# hist_feat=True
# hog_feat=True

# # Collect Data
# car_image_paths, noncar_image_paths = collect_data()
# # Split Train/Test Dataset
# X_train,X_test,y_train,y_test,X_scaler = dataset_split(car_image_paths,noncar_image_paths,
#                                             cspace,pix_per_cell,cell_per_block,orient,
#                                             hog_channel,spatial_size,hist_bins,hist_range,
#                                             spatial_feat,hist_feat,hog_feat)
# # Train Linear SVC
# svc = linearSVC_Classifier(X_train,y_train,X_test,y_test)

# hot_windows = []
# for i in range(len(x_start_stop)):
#     window_list = sliding_window(test_img,x_start_stop[i],y_start_stop[i],xy_window[i],xy_overlap[i])
#     hot_windows.append(search_windows(test_img,window_list,svc,X_scaler,cspace,pix_per_cell,
#                     cell_per_block,orient,hog_channel,spatial_size,hist_bins,
#                     hist_range,spatial_feat,hist_feat,hog_feat))

# hot_windows = [item for sublist in hot_windows for item in sublist]

# win_img = draw_box(test_img,hot_windows)

# plt.imshow(win_img)
# plt.show()

########################
# Draw Labeled boxes

image = cv2.imread('./test_images/test6.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
test_img = np.copy(image)

window_image = process_image(test_img)

plt.imshow(window_image)
plt.show()