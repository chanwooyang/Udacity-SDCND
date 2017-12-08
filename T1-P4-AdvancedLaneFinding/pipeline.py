from calibrateCamera import *
from threshold import *
from laneDetector import *
from laneDrawer import *
from line import *

# import imageio
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip

objpoints, imgpoints = findCBCorners()
abs_sobel_thr = [20,140]
abs_sobel_ksize = 5
mag_thr = [90,130]
mag_ksize = 3
dir_thr = [np.pi/4,np.pi/3]
dir_ksize = 15
hls_thr = [180,255]

def process_image(image):
	image, undistorted_img = correctDistortion(image, objpoints, imgpoints)
	thresh_image = combined_threshold(undistorted_img,abs_sobel_thr,mag_thr,dir_thr,hls_thr,abs_sobel_ksize,mag_ksize,dir_ksize)
	thresh_img, warped_img, M, M_inv, src, dst = perspectiveTransform(thresh_image)

	nonzero = warped_img.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])

	if Left.detected == False or Right.detected == False:
		left_lane, right_lane, ploty, _ = slidingWindowLaneSearch(warped_img)

		# Check distance between Left and Right lanes
		Left.recent_xfitted = left_lane[2]
		Right.recent_xfitted = right_lane[2]
		x_fitted_diff = Right.recent_xfitted - Left.recent_xfitted

		if x_fitted_diff > 350 and x_fitted_diff < 550:
			# Unfold Left Lane Data
			Left.index = left_lane[0]
			Left.current_fit = left_lane[1]
			Left.detected = True
			# Unfold Right Lane Data
			Right.index = right_lane[0]
			Right.current_fit = right_lane[1]
			Right.detected = True
		else:	# BAD x fitted value: Do not update Lane.current_fit
			pass

	elif Left.detected == True and Right.detected == True:
		left_lane2, right_lane2, _ = prevFrameLaneSearch(warped_img,Left.index,Right.index,Left.best_fit,Right.best_fit)
		
		# Unfold Left Lane Data
		Left.index = left_lane2[0]
		Left.current_fit = left_lane2[1]
		left_fy = left_lane2[2]
		# Unfold Right Lane Data
		Right.index = right_lane2[0]
		Right.current_fit = right_lane2[1]
		right_fy = right_lane2[2]


	if len(Left.current_fit) == 3  and len(Right.current_fit) == 3:
		
		# Check if new fit is close to last fit
		if Left.best_fit != None and Right.best_fit != None:
			Left.diffs = Left.best_fit - Left.current_fit
			Right.diffs = Right.best_fit - Right.current_fit
		else:
			pass

		if Left.diffs[0]>0.0005 or Left.diffs[1] > 0.5 or Left.diffs[2] > 50:
			Left.reset()
		else:	
			# Update Best Fit
			Left.best_fit = Left.ewma_filter(Left.best_fit,Left.current_fit,0.8)
		
		if Right.diffs[0]>0.0005 or Right.diffs[1] > 0.5 or Right.diffs[2] > 50:
			Right.reset()
		else:	
			# Update Best Fit
			Right.best_fit = Right.ewma_filter(Right.best_fit,Right.current_fit,0.8)

		# Draw Lane on image frame
		lane_img = drawLane(image,warped_img,Left.best_fit,Right.best_fit,M_inv)
		# Compute Curve Radius
		left_R_curve, right_R_curve = computeCurvature(warped_img,Left.index,Right.index)

		Left.radius_of_curvature = Left.ewma_filter(Left.radius_of_curvature,left_R_curve,0.95)

		Right.radius_of_curvature = Right.ewma_filter(Right.radius_of_curvature,right_R_curve,0.95)

		R_curve_avg = (Left.radius_of_curvature+Right.radius_of_curvature)/2
		# Compute offset from center
		center_offset = computeCenterOffset(image,Left.best_fit,Right.best_fit)

		Left.line_base_pos = Left.ewma_filter(Left.line_base_pos,center_offset,0.95)

		# Draw data info on image frame
		lane_data_final_img = drawData(lane_img,R_curve_avg,Left.line_base_pos)

		return lane_data_final_img
	else: 
		try:
			# Draw Lane on image frame
			lane_img = drawLane(image,warped_img,Left.best_fit,Right.best_fit,M_inv)
			R_curve_avg = (Left.radius_of_curvature+Right.radius_of_curvature)/2
			# Draw data info on image frame
			lane_data_final_img = drawData(lane_img,R_curve_avg,Left.line_base_pos)

			return lane_data_final_img
		except:
			return image

Left = Line()
Right = Line()

# # path = 'test_images/straight_lines2.jpg'
# path = 'test_images/test3.jpg'
# image = cv2.imread(path)
# output = process_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
# plt.imshow(output)
# plt.show()


if not os.path.exists('test_videos_output/'):
    os.makedirs('test_videos_output/')
video_output1= 'test_videos_output/processed_project_video.mp4'
# video_output2 = 'test_videos_output/processed_challenge_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
# clip2 = VideoFileClip("challenge_video.mp4")
video_process = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)
video_process.write_videofile(video_output1, audio=False)



