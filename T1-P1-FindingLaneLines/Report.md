# **Finding Lane Lines on the Road** 

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline function, 'lane_finding_pipeline(image)' consists of 6 steps. First, I changed the image into the grayscale, and then Gaussian Smoothing was applied with kernel_size = 5. Then, Canny Edge was applied, and the mask was applied to the region of interest. Lastly, Masked Canny Edge image was fed into the Hough Transform function to draw the line. Then, this line image was superimposed on the original image by using weighted_img function.

'draw_lines()' function, first, was changed such that it separates the lines on the left lane and those on the right lane by determining whether the slope of each line is negative or positive and also whether the line is on left side or right side of an image. Then, these x-points and y-points are collected in different list, and then they were fed into the numpy.polyfit, to find coefficients of linear regression fit function, which are slope and x-intercept. Slope and x-intercept points are, then, fed into 'ewma_filter()' function that I created to reduce the noise in line drawing. Its outputs are used to find x and y-coordinates to draw extrapolated lines on each left and right lanes.

To see how the pipeline works, there is 'plt.imshow()' function at the end of each step. Uncomment this to show the image process result after each step.


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming is when a car drives on a curved lane. Sicne 'draw_line()' function draws a linear line, it cannot draw a nicely fitting curved line. Also, the way 'draw_line()' determine whether a line is for left lane or right lane is by checking its slope value and position on the entire image (left side or right side). However, this method is not valid anymore on the image with sharp-curved road.

Another potential shortcoming is when there are shades on a road. This will creates bunch of edges in many different directions right in front of the car, so this will introduce great amount of noise into the 'draw_line()' algorithm.


### 3. Suggest possible improvements to your pipeline

'draw_line()' function should be updated in a way that it draws a non-linear line instead of a linear line.
Also, more techniques need to be introduced to reject all noises created by every object other than lanes (e.g. shades, edge of car, crack on road, etc.)
