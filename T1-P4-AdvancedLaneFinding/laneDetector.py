from calibrateCamera import *
from threshold import *
import matplotlib.pyplot as plt

def slidingWindowLaneSearch(image):

	# Histogram of portion of an image
	hist = np.sum(image[image.shape[0]//2:,:], axis=0)
	# plt.plot(hist)
	# plt.show()

	# Find the peak of the left and right halves of the histogram
	midpoint = np.int(image.shape[1]/2)
	quarterpoint = np.int(midpoint/2)
	leftx_base = np.argmax(hist[quarterpoint:midpoint])+quarterpoint
	rightx_base = np.argmax(hist[midpoint:midpoint+quarterpoint])+midpoint
	# Choose number of sliding windows
	num_windows = 9
	# Height of sliding windows
	window_height = np.int(image.shape[0]/num_windows)
	# Identify x and y positions of all nonzero pixels in the image
	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	# Current positions to be updated for the each window
	leftx_current = leftx_base
	rightx_current = rightx_base
	# Set the width of windows +/- margin
	margin = 40
	# Set min number of pixels found to recenter window
	min_pixel = 70
	# Create empty lists to receive left and right lane pixel indices
	left_lane_idx = []
	right_lane_idx = []
	# Create an empty list to store each window vertices for data-visualization purpose
	window_vertices = []
	# Step through windows one by one
	for window in range(num_windows):
		# Identify window boundaries in x and y (and right and left)
		win_y_low = image.shape[0] - (window+1)*window_height
		win_y_high = image.shape[0] - window*window_height
		win_xleft_low = leftx_current - margin
		win_xleft_high = leftx_current + margin
		win_xright_low = rightx_current - margin
		win_xright_high = rightx_current + margin
		# Append window vertices
		window_vertices.append((win_y_low,win_y_high,win_xleft_low,win_xleft_high,win_xright_low,win_xright_high))
		# Identify the nonzero pixels in x and y within the window
		good_left_idx = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xleft_low)&
			(nonzerox<win_xleft_high)).nonzero()[0]
		good_right_idx = ((nonzeroy>=win_y_low)&(nonzeroy<win_y_high)&(nonzerox>=win_xright_low)&
			(nonzerox<win_xright_high)).nonzero()[0]
		# Appned identified indices to lists
		left_lane_idx.append(good_left_idx)
		right_lane_idx.append(good_right_idx)
		# If window > min_pixel, recenter next window on its mean position
		if len(good_left_idx) > min_pixel:
			leftx_current = np.int(np.mean(nonzerox[good_left_idx]))
		if len(good_right_idx) > min_pixel:
			rightx_current = np.int(np.mean(nonzerox[good_right_idx]))
	# Concatenate arrays of indices
	left_lane_idx = np.concatenate(left_lane_idx)
	right_lane_idx = np.concatenate(right_lane_idx)

	# Extract left and right lane pixel positions
	leftx = nonzerox[left_lane_idx]
	lefty = nonzeroy[left_lane_idx]
	rightx = nonzerox[right_lane_idx]
	righty = nonzeroy[right_lane_idx]

	# Fit a second order polynomial
	left_fit = np.polyfit(lefty,leftx,2)
	right_fit = np.polyfit(righty,rightx,2)

	# Generate x and y values for plotting
	# f(y) = Ay^2 + By + C
	ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
	left_fy = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fy = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

	left_lane = (left_lane_idx,left_fit,leftx_current)
	right_lane = (right_lane_idx,right_fit,rightx_current)

	return left_lane, right_lane, ploty, window_vertices


def prevFrameLaneSearch(image,left_lane_idx,right_lane_idx,left_fit,right_fit):
	
	nonzerox = np.array(image.nonzero()[1])
	nonzeroy = np.array(image.nonzero()[0])

	margin = 50
	# Left: f(y) = Ay^2 + By + C
	left_fy = left_fit[0]*(nonzeroy**2)+left_fit[1]*nonzeroy+left_fit[2]
	left_lane_idx = ((nonzerox > left_fy-margin) & (nonzerox < left_fy+margin))
	# Right: f(y) = Ay^2 + By + C
	right_fy = right_fit[0]*(nonzeroy**2)+right_fit[1]*nonzeroy+right_fit[2]
	right_lane_idx = ((nonzerox > right_fy-margin) & (nonzerox < right_fy+margin))

	# Extract left and right lane pixel positions
	leftx = nonzerox[left_lane_idx]
	lefty = nonzeroy[left_lane_idx]
	rightx = nonzerox[right_lane_idx]
	righty = nonzeroy[right_lane_idx]

	#Fit a second order polynomial
	left_fit = np.polyfit(lefty,leftx,2)
	right_fit = np.polyfit(righty,rightx,2)

	left_lane = (left_lane_idx,left_fit,left_fy)
	right_lane = (right_lane_idx,right_fit,right_fy)

	return left_lane, right_lane, margin


ym_per_pixel = 30./720	# Meter per pixel in y-dimension
xm_per_pixel = 3.7/700	# Meter per pixel in x-dimension

# ym_per_pixel = 3.048/100	# Meter per pixel in y-dimension
# xm_per_pixel = 3.7/378	# Meter per pixel in x-dimension
def computeCurvature(image,left_lane_idx,right_lane_idx):
	# Identify x and y positions of all nonzero pixels in the image
	nonzero = image.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	# Extract left and right lane pixel positions
	leftx = nonzerox[left_lane_idx]
	lefty = nonzeroy[left_lane_idx]
	rightx = nonzerox[right_lane_idx]
	righty = nonzeroy[right_lane_idx]

	# Convert from pixel to meter
	leftx_meter = leftx*xm_per_pixel
	lefty_meter = lefty*ym_per_pixel
	rightx_meter = rightx*xm_per_pixel
	righty_meter = righty*ym_per_pixel

	# Fit a second order polynomial
	left_fit_meter = np.polyfit(lefty_meter,leftx_meter,2)
	right_fit_meter = np.polyfit(righty_meter,rightx_meter,2)
	# f(y) = Ay^2 + By + C
	# Left Lane Polyfit Coefficients
	A_left = left_fit_meter[0]
	B_left = left_fit_meter[1]
	C_left = left_fit_meter[2]
	# Right Lane Polyfit Coefficients
	A_right = right_fit_meter[0]
	B_right = right_fit_meter[1]
	C_right = right_fit_meter[2]

	ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
	y_eval = np.max(ploty)
	# f'(y) = dx/dy = 2Ay + B
	# Left Lane
	left_dfy = 2*A_left*y_eval + B_left
	# Right Lane
	right_dfy = 2*A_right*y_eval + B_right

	# f''(y) = ddx/dy^2 = 2A
	# Left Lane
	left_ddfy = 2*A_left
	# Right Lane
	right_ddfy = 2*A_right

	# Raidus of Curvature in pixel
	# R_curve = ((1+(dx/dy)^2)^(3/2))/|ddx/dy^2|
	# Left Lane Radius of Curvature
	left_R_curve = ((1+(left_dfy**2))**(3/2))/np.absolute(left_ddfy)
	# Right Lane Radius of Curvature
	right_R_curve = ((1+(right_dfy**2))**(3/2))/np.absolute(right_ddfy)

	return left_R_curve, right_R_curve

def computeCenterOffset(image,left_fit,right_fit):

	# f(y) = Ay^2 + By + C
	# Left Lane Polyfit Coefficients
	A_left = left_fit[0]
	B_left = left_fit[1]
	C_left = left_fit[2]
	# Right Lane Polyfit Coefficients
	A_right = right_fit[0]
	B_right = right_fit[1]
	C_right = right_fit[2]

	image_bottom = image.shape[0]
	car_position = image.shape[1]/2
	left_fy_x = A_left*(image_bottom**2)+B_left*image_bottom+C_left
	right_fy_x = A_right*(image_bottom**2)+B_right*image_bottom+C_right
	center_position = (left_fy_x+right_fy_x)/2
	center_offset = (car_position - center_position)*xm_per_pixel

	return center_offset




