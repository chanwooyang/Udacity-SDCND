## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames.  If you want to extract more test images from the videos, you can simply use an image writing method like `cv2.imwrite()`, i.e., you can read the video in frame by frame as usual, and for frames you want to save for later you can write to an image file.  

To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include a description in your writeup for the project of what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

The `challenge_video.mp4` video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions.  The `harder_challenge.mp4` video is another optional challenge and is brutal!

If you're feeling ambitious (again, totally optional though), don't stop there!  We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

---


# Advanced Lane Finding Project: Writeup Report

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/CBCorner.png "CBCorner"
[image2]: ./output_images/CBUndistorted.png "CBUndistorted"
[image3]: ./output_images/TestUndistorted.png "TestUndistorted"
[image4]: ./output_images/TestWarped.png "TestWarped"
[image5]: ./output_images/StraightWarped.png "StraightWarped"
[image6]: ./output_images/GradBinary.png "GradBinary"
[image7]: ./output_images/MagBinary.png "MagBinary"
[image8]: ./output_images/DirBinary.png "DirBinary"
[image9]: ./output_images/HLSBinary.png "HLSBinary"
[image10]: ./output_images/CombinedBinary.png "CombinedBinary"
[image11]: ./output_images/slidingWindowLaneSearch.png "slidingWindowLaneSearch"
[image12]: ./output_images/prevFameLaneSearch.png "prevFameLaneSearch"
[image13]: ./output_images/Equation.png "Equation"
[image14]: ./output_images/pipeline_image.png "pipeline_image"


## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how the camera matrix and distortion coefficients were computed.

The code for this step is contained in the 'calibrateCamera.py' file.

As a first step of the camera calibration, 'findCBCorners()' funciton was created to find chessboard corners. Chessboard images taken from different angles in 'camera_cal' folder were used for this process.

It was started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here, it was assumed that the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoint` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

![Find Chess Board Corner Example][image1]

Then, in 'correctDistortion()' function, these `objpoints` and `imgpoints` were used to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. These coefficients were used to undistort camera images using the `cv2.undistort()` 

![Undistorted Chess Board Image Example][image2]


### Pipeline

#### 1. Example of a distortion-corrected image.

Using 'findCBCorners()' and 'correctDistortion()' functions in 'calibrateCamera.py' file, test images were also corrected. 
![Corrected Test Images][image3]

#### 2. Describe how you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the 'calibrateCamera.py' file.

After a camera image was undistorted, the perspective of the image was transformed (or warped) by computing the transform matrix using 'cv2.getPerspectiveTransform(src,dst)'. The inverse matrix was also computed for unwarped a processed image back to the original perspective later.  It was decided to use hardnumbers for the source and destination points in the following manner:

```python
# [Bot Left], [Bot Right], [Top Left], [Top Right]
offset_x = 400
offset_y = 100
src = np.float32([[266, 680],[1042, 680],[567, 470],[717, 470]])
dst = np.float32([[offset_x, img_size[1]],[img_size[0]-offset_x, img_size[1]],
				[offset_x, offset_y],[img_size[0]-offset_x, offset_y]])
```

Hardnumber values were manually adjusted by visually checking the lines drawn on the undistorted image and final warped image. 

Then, finally, `cv2.warpPerspective()` was used to create so-called 'bird-eye view' of an image.

![Wapred Image][image4]

It was verified that the perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Straight Lane Warped Image][image5]

#### 3. Describe how you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

A combination of color and gradient thresholds was used to generate a binary image: `abs_sobel_threshold()` `mag_threshold()`, `dir_threshold()`, and `hls_threshold()` functions in `threshold.py` file. Each of binary images were combined in the following manner:
```python
def combined_threshold(image,abs_sobel_thr,mag_thr,dir_thr,hls_thr,abs_sobel_ksize,mag_ksize,dir_ksize):

	gradxy = abs_sobel_threshold(image,abs_sobel_ksize,abs_sobel_thr)
	mag_binary = mag_threshold(image,mag_ksize,mag_thr)
	dir_binary = dir_threshold(image,dir_ksize,dir_thr)
	hls_binary = hls_threshold(image,hls_thr)

	combined_binary = np.zeros_like(dir_binary)
	combined_binary[((gradxy==1)|(hls_binary==1))|(mag_binary==1)&(dir_binary==1)] = 1

	return combined_binary
```

Threshold values were chosen by repeating trial and error process until it generates acceptable images.

![Absolute Sobel Threshold Image][image6]
![Magnitued Threshold Image][image7]
![Direction Threshold Image][image8]
![HLS Color Space Threshold Image][image9]
![Combined Threshold Image][image10]

#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the 'landDetector.py' file.

First, `slidingWindowLaneSearch()` function was implemented to identify lane-line pixels and lane-line polynomial fits based on finding peaks of histogram of an image. A pixel histogram of a bottom half of an input image was computed. Margin of quarter point of an image width was also taken on both left and right side of image to focus on lane-line part. From this histogram, two peak points were considered to be the base of left and right lanes. Then, search windows were created to identify  lane-line pixels. For every search, if number of identified lane-line pixels are greater than that of previously found lane-line position, then the lane line poisiton is readjusted. All these identified lane-line indices were collected and are used to extract lane-line positions. Once both left and right lane-line positions were extracted, then `np.polyfit()` was used to find a second degree polynomial fit for identified lane-line pixel positions. 

![slidingWindowLaneSearch image][image11]

Next, `prevFrameLaneSearch()` function was implemented to identify lane-line pixels based on lane-line polynomial fits computed from a previous frame. This function used a polynomial fit and upate lane-line pixel indices within margin, and computed new a polynomial fit using the updated lane-line pixel indices. As this function did not need to use histogram and sliding window search, it simplified and made the lane detection process faster.

![prevFrameLaneSearch image][image12]

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the 'landDetector.py' file.

The radius of curvature of lane was computed by `computeCurvature()` function. Radius of curvature function was implemented based on the given equation:

![Radius of Curvature Equation][image13]

The position of vehicle with respect to center was computed by `computeCenterOffset()` function. In this function, it was assumed that the camera was mounted exactly at the center of the car, so the center of image was considered to be the center of the car. Then, based on input left and right lane-line fits, an avarage was computed to find a mid point between left and right lanes, and this was compared with the center point of the image.

For both radius of curvature and vehicle positions, pixel-to-meter converted polynomial fits were used compute those values.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the 'landDrawer.py' file.

`drawLane()` function drew lane-lines for both left and right lanes based on input polynomial fits. Once lanes were drawn, the lane image was unwarped back to the original perspective using the same function, `cv2.warpPerspective()`, but with perspective matrix inverse.

`drawData()` function drew the computed radius of curvature value and the vehicel position value.

![Pipeline Image][image14]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code for this step is contained in the 'pipeline.py' file.

Using the 'Line' Class in `line.py` file, left lane object and right lane were instantiated to keep track of the lane characteristics between video frames. Mainly each object kept track of lane-line polynomial fit, radius of curvature, and vehicle position. If a newly computed polynomial fit from the current image frame was too deviated from the saved best lane-line polynomial fit, then the function did not update the saved lane-line polynomial fit, but instead, it relied on previously saved best data to draw lanes and to display radius of curvature and vehicel position values. Also, it reset the lane-detection boolean, and searched for lane-line pixels, again, with `slidingWindowLaneSearch()` method. Once lanes were detected, again, and new polynomial fits were close enough to the best fit, then the function used `prevFrameLaneSearch()` method to detect lanes and updated object attributes: best fit, radius of curvature, vehicle position, etc.

Final polynomial fit, radius of curvature, and vehicle position values for the final processed video were computed using EWMA filter to reduce high frequency noise.

Here's a [link to my video result](test_videos_output/processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. It was observed that sudden change in brightness affected the lane detection. This could be because the threshold image produced by the current method was not robust to enough to handle this.

2. If there were line or color-transition line between left and right lanes (like in the challenge video), then this also caused an error on the lane detection. Since the left and right lane base detection relied on the find maximum peak values from the histogram, a line between left and right lanes could result the highest peak in the histogram.

3. When an undistorted image was transformed to create a 'bird-eye' view, there was an assumption that the road was flat, and hardnumbers were used for 'src' and 'dst'. Thus, if there were uphill or downhill images, then the function incorrectly transforemd the image.

#### 2. Future Improvement

1. Though current implementation was good enough as the system could rely on previously saved best fit data, threshold combining function could be improved to handle the sudden change in brightness. Or, different threshold combining function could be introduced as another option for different brightness of an image.

2. Since left lane and right lane were mostly separated at a same distance, few extra lines of code could be implemented in such way that it compares pixel distance between two maximum peak of histogram and if the distance is not reasonable, then it finds next highest peak of histogram and compare the distance. 

3. Instead of hardnumbers, 'src' and 'dst' values should be dynamically computed based on the gradient (or slope) of the road.