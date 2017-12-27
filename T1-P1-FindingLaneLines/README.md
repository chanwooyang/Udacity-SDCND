# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


Creating a Great Writeup
---
For this project, a great writeup should provide a detailed response to the "Reflection" section of the [project rubric](https://review.udacity.com/#!/rubrics/322/view). There are three parts to the reflection:

1. Describe the pipeline

2. Identify any shortcomings

3. Suggest possible improvements

We encourage using images in your writeup to demonstrate how your pipeline works.  

All that said, please be concise!  We're not looking for you to write a book here: just a brief description.

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup. Here is a link to a [writeup template file](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md). 


The Project
---

## If you have already installed the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) you should be good to go!   If not, you should install the starter kit to get started on this project. ##

**Step 1:** Set up the [CarND Term1 Starter Kit](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook

You will complete the project code in a Jupyter notebook.  If you are unfamiliar with Jupyter Notebooks, check out <A HREF="https://www.packtpub.com/books/content/basics-jupyter-notebook-and-python" target="_blank">Cyrille Rossant's Basics of Jupyter Notebook and Python</A> to get started.

Jupyter is an Ipython notebook where you can run blocks of code and see results interactively.  All the code for this project is contained in a Jupyter notebook. To start Jupyter in your browser, use terminal to navigate to your project directory and then run the following command at the terminal prompt (be sure you've activated your Python 3 carnd-term1 environment as described in the [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) installation instructions!):

`> jupyter notebook`

A browser window will appear showing the contents of the current directory.  Click on the file called "P1.ipynb".  Another browser window will appear displaying the notebook.  Follow the instructions in the notebook to complete the project.  

**Step 3:** Complete the project and submit both the Ipython notebook and the project writeup

------------------------

# Finding Lane Lines on the Road

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

