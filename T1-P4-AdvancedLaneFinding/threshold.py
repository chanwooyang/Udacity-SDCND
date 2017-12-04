import cv2
import numpy as np

def abs_sobel_threshold(image,ksize,threshold):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize)
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize)
	
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)

	scaled_sobelx = (255.0*abs_sobelx/np.max(abs_sobelx)).astype(np.uint8)
	scaled_sobely = (255.0*abs_sobely/np.max(abs_sobely)).astype(np.uint8)

	binary_output = np.zeros_like(scaled_sobelx)
	binary_output[(scaled_sobelx>threshold[0]) & (scaled_sobelx<threshold[1]) &
				 (scaled_sobely>threshold[0]) & (scaled_sobely<threshold[1])] = 1

	return binary_output

def mag_threshold(image,ksize,threshold):
	# Convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize)
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize)

	abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)
	scaled_sobelxy = (255.0*abs_sobelxy/np.max(abs_sobelxy)).astype(np.uint8)

	binary_output = np.zeros_like(scaled_sobelxy)
	binary_output[(scaled_sobelxy>threshold[0]) & (scaled_sobelxy<threshold[1])] = 1

	return binary_output

def dir_threshold(image,ksize,threshold):
	# Convert to grayscale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize)
	sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize)

	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)

	graddir = np.arctan2(abs_sobely,abs_sobelx)

	binary_output = np.zeros_like(graddir)
	binary_output[(graddir>threshold[0]) & (graddir<threshold[1])] = 1

	return binary_output

def hls_threshold(image,threshold):
	hls = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
	s_channel = hls[:,:,2]	# Saturation Channel
	
	binary_output = np.zeros_like(s_channel)
	binary_output[(s_channel>threshold[0]) & (s_channel<threshold[1])] = 1

	return binary_output

def combined_threshold(image,abs_sobel_thr,mag_thr,dir_thr,hls_thr,abs_sobel_ksize,mag_ksize,dir_ksize):

	gradxy = abs_sobel_threshold(image,abs_sobel_ksize,abs_sobel_thr)
	mag_binary = mag_threshold(image,mag_ksize,mag_thr)
	dir_binary = dir_threshold(image,dir_ksize,dir_thr)
	hls_binary = hls_threshold(image,hls_thr)

	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((gradxy==1)|(hls_binary==1))|(mag_binary==1)&(dir_binary==1)] = 1

	return combined_binary

