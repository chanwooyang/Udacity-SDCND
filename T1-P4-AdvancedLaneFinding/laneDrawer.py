from calibrateCamera import *
from threshold import *
from laneDetector import *

def drawLane(image,warped_img,left_fit,right_fit,M_inv):
	new_img = np.copy(image)
	# Create an image to draw the lines on
	warp_zero = np.zeros_like(warped_img).astype(np.uint8)
	color_warp = np.dstack((warp_zero, warp_zero, warp_zero))


	ploty = np.linspace(0,image.shape[0]-1,image.shape[0])
	left_fy = left_fit[0]*(ploty**2)+left_fit[1]*ploty+left_fit[2]
	right_fy = right_fit[0]*(ploty**2)+right_fit[1]*ploty+right_fit[2]
	# Recast the x and y points into usable format for cv2.fillPoly()
	pts_left = np.array([np.transpose(np.vstack([left_fy, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fy, ploty])))])
	pts = np.hstack((pts_left, pts_right))

	# Draw the lane onto the warped blank image
	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
	cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
	cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

	# Warp the blank back to original image space using inverse perspective matrix (Minv)
	newwarp = cv2.warpPerspective(color_warp, M_inv, (image.shape[1], image.shape[0])) 
	# Combine the result with the original image
	result = cv2.addWeighted(new_img, 1, newwarp, 0.3, 0)

	return result

def drawData(image,curve_radius,center_offset):
	new_img = np.copy(image)
	font = cv2.FONT_HERSHEY_SIMPLEX
	radius_text = 'Curve Radius: {0:.3f} m'.format(curve_radius)
	cv2.putText(new_img,radius_text,(40,70),font,1.5,(200,100,100),2,cv2.LINE_AA)

	if center_offset > 0:
		direction  = 'RIGHT'
	elif center_offset < 0:
		direction = 'LEFT'
	else:
		direction = ''

	abs_center_offset = np.absolute(center_offset)
	center_offset_text = '{0:.3f} m {1} from center'.format(abs_center_offset,direction)
	cv2.putText(new_img,center_offset_text,(40,120),font,1.5,(200,100,100),2,cv2.LINE_AA)

	return new_img





