import os
import cv2
import numpy as np

def findCBCorners():
	path = 'camera_cal/'

	# Number of corners
	nx = 9
	ny = 6

	# Prepare Object Points
	obj_points = np.zeros((ny*nx,3), np.float32)
	# print(obj_points)
	obj_points[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
	# print(obj_points)

	# Arrays to store object points and image points from all images
	objpoints = []
	imgpoints = []

	for cal_img in os.listdir(path):
		if cal_img.startswith('calibration'):
			img = cv2.imread(path+cal_img)
			gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray_img, (nx,ny), None)

			if ret == True:
				objpoints.append(obj_points)
				imgpoints.append(corners)

				# Draw and display the corners
				chess_img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
				
				# # Display Corners on chessboard images
				# cv2.imshow('chessboard corner', chess_img)
				# cv2.waitKey(0)
			else:
				pass
		else:
			pass

	assert(len(objpoints) == len(imgpoints))

	return objpoints, imgpoints


def drawLines(img, vertices):
	color=[0, 0, 250]
	thickness=3
	# Vertices: [Bot Left], [Bot Right], [Top Left], [Top Right]
	cv2.line(img, tuple(vertices[0]),tuple(vertices[1]),color,thickness)
	cv2.line(img, tuple(vertices[1]),tuple(vertices[3]),color,thickness)
	cv2.line(img, tuple(vertices[3]),tuple(vertices[2]),color,thickness)
	cv2.line(img, tuple(vertices[2]),tuple(vertices[0]),color,thickness)


def correctDistortion(image, objpoints, imgpoints):
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)
	undistorted_img = cv2.undistort(image, mtx, dist, None, mtx)
	return image, undistorted_img


def perspectiveTransform(undist_image):
	img_size = (undist_image.shape[1], undist_image.shape[0])	#(1280,720)

	# [Bot Left], [Bot Right], [Top Left], [Top Right]
	offset_x = 400
	offset_y = 100
	src = np.float32([[266, 680],[1042, 680],[567, 470],[717, 470]])
	dst = np.float32([[offset_x, img_size[1]],[img_size[0]-offset_x, img_size[1]],
		[offset_x, offset_y],[img_size[0]-offset_x, offset_y]])
	# print(dst)

	# Draw Lines
	# drawLines(undist_image, src)

	# Tranform Matrix
	M = cv2.getPerspectiveTransform(src,dst)
	# Inverse Transform Matrix
	M_inv = cv2.getPerspectiveTransform(dst,src)

	warped_img = cv2.warpPerspective(undist_image, M, img_size)

	return undist_image, warped_img, M, M_inv, src, dst
	
