from calibrateCamera import *
from threshold import *
from laneDetector import *
import matplotlib.pyplot as plt

# path = 'test_images/straight_lines2.jpg'
path = 'test_images/test1.jpg'
image = cv2.imread(path)

################################################################
#Lane Dectection Visualization
################################################################
abs_sobel_thr = [20,140]
abs_sobel_ksize = 5
mag_thr = [90,130]
mag_ksize = 3
dir_thr = [np.pi/4,np.pi/3]
dir_ksize = 15
hls_thr = [180,255]

objpoints, imgpoints = findCBCorners()
image, undistorted_img = correctDistortion(image, objpoints, imgpoints)
thresh_image = combined_threshold(undistorted_img,abs_sobel_thr,mag_thr,dir_thr,hls_thr,abs_sobel_ksize,mag_ksize,dir_ksize)
thresh_img, warped_img, M, M_inv, src, dst = perspectiveTransform(thresh_image)

nonzero = warped_img.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])


# ######### Sliding Window Search ##################################################
left_lane, right_lane, ploty, window_vertices = slidingWindowLaneSearch(warped_img)

# # Unfold Left Lane Data
# left_lane_idx = left_lane[0]
# left_fit = left_lane[1]
# left_fy = left_lane[2]
# # Unfold Right Lane Data
# right_lane_idx = right_lane[0]
# right_fit = right_lane[1]
# right_fy = right_lane[2]

# # Create an output image to draw on and visualize the result
# out_img = np.uint8(np.dstack((warped_img,warped_img,warped_img))*255)

# # Draw the windows on the visualization image
# for vertex in window_vertices:
# 	cv2.rectangle(out_img,(vertex[2],vertex[0]),(vertex[3],vertex[1]),(0,255,0),2)
# 	cv2.rectangle(out_img,(vertex[4],vertex[0]),(vertex[5],vertex[1]),(0,255,0),2)


# out_img[nonzeroy[left_lane_idx],nonzerox[left_lane_idx]]=[255,0,0]
# out_img[nonzeroy[right_lane_idx],nonzerox[right_lane_idx]]=[0,0,255]
# plt.imshow(out_img)
# plt.plot(left_fy,ploty,color='yellow')
# plt.plot(right_fy,ploty,color='yellow')
# plt.xlim(0,1280)
# plt.ylim(720,0)
# plt.show()

################################################################
######## Previous Frame Lane Search#############################
left_lane2, right_lane2, margin = prevFrameLaneSearch(warped_img, left_lane, right_lane)

# Unfold Left Lane Data
left_lane_idx2 = left_lane2[0]
left_fit2 = left_lane2[1]
# Unfold Right Lane Data
right_lane_idx2 = right_lane2[0]
right_fit2 = right_lane2[1]
# Generate x and y values for plotting
left_fy2 = left_fit2[0]*(ploty**2)+left_fit2[1]*ploty+left_fit2[2]
right_fy2 = right_fit2[0]*(ploty**2)+right_fit2[1]*ploty+right_fit2[2]
# Create an output image to draw on and visualize the result
out_img = np.uint8(np.dstack((warped_img,warped_img,warped_img))*255)
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_idx2],nonzerox[left_lane_idx2]] = [255,0,0]
out_img[nonzeroy[right_lane_idx2],nonzerox[right_lane_idx2]] = [0,255,0]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
print(left_fy2.shape)
print(ploty.shape)
left_line_window1 = np.array([np.transpose(np.vstack([left_fy2-margin,ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fy2+margin,ploty])))])
left_line_pts = np.hstack((left_line_window1,left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fy2-margin,ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fy2+margin,ploty])))])
right_line_pts = np.hstack((right_line_window1,right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]),(0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]),(0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fy2, ploty, color='yellow')
plt.plot(right_fy2, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)
plt.show()

left_R_curve, right_R_curve = computeCurvature(warped_img,left_lane_idx2,right_lane_idx2)
center_offset = computeCenterOffset(image,left_fit2,right_fit2)
print('{0:.3f} m'.format(left_R_curve))
print('{0:.3f} m'.format(right_R_curve))
print('{0:.3f} m'.format(center_offset))


################################################################

# abs_sobel_thr = [20,140]
# abs_sobel_ksize = 5
# mag_thr = [90,130]
# mag_ksize = 3
# dir_thr = [np.pi/4,np.pi/3]
# dir_ksize = 15
# hls_thr = [180,255]

# grad_binary = abs_sobel_threshold(image,abs_sobel_ksize,abs_sobel_thr)
# mag_binary = mag_threshold(image,mag_ksize,mag_thr)
# dir_binary = dir_threshold(image,dir_ksize,dir_thr)
# hls_binary = hls_threshold(image,hls_thr)
# combined_binary = combined_threshold(image,abs_sobel_thr,mag_thr,dir_thr,hls_thr,abs_sobel_ksize,mag_ksize,dir_ksize)
# # Plot the result
# plt.subplot(121),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),plt.title('Original Image')
# plt.subplot(122),plt.imshow(combined_binary,'gray'),plt.title('Threshold Image')
# plt.show()


################################################################
# # Show warped image
# objpoints, imgpoints = findCBCorners()
# image, undist = correctDistortion(image, objpoints, imgpoints)
# undist_image, warped_image, M, M_inv, src, dst = perspectiveTransform(undist)

# drawLines(image, src)
# plt.subplot(121),plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)),plt.title('Original Image')
# drawLines(warped_image, dst)
# plt.subplot(122),plt.imshow(cv2.cvtColor(warped_image,cv2.COLOR_BGR2RGB)),plt.title('Warped Image')
# plt.show()


################################################################
## Show undistorted image
# path = 'camera_cal/'
# objpoints, imgpoints = findCBCorners()
# for cal_img in os.listdir(path):
# 		if cal_img.startswith('calibration'):
# 			cal_image = cv2.imread(path+cal_img)
# 			img, undistorted = correctDistortion(cal_image, objpoints, imgpoints)

# 			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
# 			f.tight_layout()
# 			ax1.imshow(img)
# 			ax1.set_title('Original Image', fontsize=50)
# 			ax2.imshow(undistorted)
# 			ax2.set_title('Undistorted Image', fontsize=50)
# 			plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
# 			plt.show()

