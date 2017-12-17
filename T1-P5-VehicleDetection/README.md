# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

Creating a great writeup:
---
A great writeup should include the rubric points as well as your description of how you addressed each point.  You should include a detailed description of the code used in each step (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You can submit your writeup in markdown or use another method and submit a pdf instead.

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.  

**As an optional challenge** Once you have a working pipeline for vehicle detection, add in your lane-finding algorithm from the last project to do simultaneous lane-finding and vehicle detection!

**If you're feeling ambitious** (also totally optional though), don't stop there!  We encourage you to go out and take video of your own, and show us how you would implement this project on a new video!

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_images.png
[image2]: ./output_images/noncar_images.png
[image3]: ./output_images/hog_image.png
[image4]: ./output_images/windows1.png
[image5]: ./output_images/windows2.png
[image6]: ./output_images/windows3.png
[image7]: ./output_images/windows4.png
[image8]: ./output_images/pipeline1.png
[image9]: ./output_images/pipeline2.png
[image10]: ./output_images/pipeline3.png
[image11]: ./output_images/pipeline4.png
[image12]: ./output_images/pipeline5.png
[image13]: ./output_images/pipeline6.png
[image14]: ./output_images/heat_img1.png
[image15]: ./output_images/heat_img2.png
[image16]: ./output_images/heat_img3.png
[image17]: ./output_images/heat_img4.png
[image18]: ./output_images/heat_img5.png
[image19]: ./output_images/heat_img6.png
[image20]: ./output_images/label_img1.png
[image21]: ./output_images/label_img2.png
[image22]: ./output_images/label_img3.png
[image23]: ./output_images/label_img4.png
[image24]: ./output_images/label_img5.png
[image25]: ./output_images/label_img6.png
[image26]: ./output_images/bounding_box1.png
[image27]: ./output_images/bounding_box2.png
[image28]: ./output_images/bounding_box3.png
[image29]: ./output_images/bounding_box4.png
[image30]: ./output_images/bounding_box5.png
[image31]: ./output_images/bounding_box6.png

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how you extracted HOG features from the training images.

The code for this step is in `data_collect.py` and `feature_extraction.py` files.

In `data_collect.py`, `collect_data()` function was implemented to collect car image and noncar images, given by Udacity. 

![car image example][image1]
![noncar image example][image2]

Then, in `feature_extraction.py`, feature extracting functions were implemented: `bin_spatial()`, `color_hist()`, and `hog_feature()`. `bin_spatial()` function took an image and resized it to 32x32 and then made it as a single array. And, `color_hist()` function separated each channel of color space of an image, computed histogram, and concanated them into a single feature vector.

`hog_feature()` function used `hog()` method from skimage.feature to exract hog feature of an image. 

![hog_feature image example][image3]

Then, these feature extracting functions were combined in `extract_features()` function. In this function, each type of features were extracted and they were concanated into a list of feature vectors.

Once the feature extracting functions were implemented, they were used with given car & non-car image data to generate training and test dataset. In `dataset_split()` function, above functions were used to extract features of given image dataset, and they were splitted into training and test dataset using `sklearn.model_selection.train_test_split` with 8:2 ratio. These splitted data sets were saved as pickle file for easier access later.

#### 2. Describe how you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is in `classifier.py` file.

Linear SVC was chosen as a classifier. The classifier training time was measured and once it was trained, the classifier was saved as a pickle file for easier access later.

#### 3. Explain how you settled on your final choice of HOG parameters.

The code for this step is in `feature_extraction.py` file.

The generated training and test sets were used to train a classifier. However, those dataset were varied by hyperparameter fed to each of feature extracting functions, so this affected the performance of the classifier. So, it was necessary to fine-tune parameters to yield a better performance of the classifier. Thus, the strategy was set in two steps: frist, with all parameters were set the same, only color space of images was changed to see which color space resulted the best classifier accuracy. Then, with the best resulting color space, other paremeters were tuned to get higher classifier accuracy and also faster feature extracting time.


So, as a first step, the parameters were set to the follow:

|pixel_per_cel 	|cell_per_block 	|orient 	|hog_channel 	|spatial_size 	|
|:-------------:|:-----------------:|:---------:|:-------------:|:-------------:|
|8				|2					|9			|'All'			|(32,32)		|

Then, different datasets were generated with different colorspace and the classifier was trained. Results were the follow:

|CSPACE		|CAR_FEAT_TIME	|NONCAR_FEAT_TIME	|TRAIN_TIMES	|ACCURACY 		|
|:---------:|:-------------:|:-----------------:|:-------------:|:-------------:| 
|RGB		|93.55			|90.12				|35.61			|0.9778			|
|LUV		|94.95			|91.46				|10.68			|0.9885			|
|HSV		|94.99			|92.51				|32.88			|0.9882			|
|YUV		|93.21			|89.83				|28.31			|0.9899			|
|HLS		|94.83			|92.98				|35.74			|0.9842			|
|YCrCb		|100.05			|97.71				|32.59			|0.9873			|

According to the result, 'YUV' color space resulted the best accuracy.

Then, color space was set to 'YUV', and other parameters were tuned for better accuracy and faster feature extracting time. Few experiments were executed and results were the follow:

|PIXEL_PER_CELL	|ORIENT	|CAR_FEAT_TIME	|NONCAR_FEAT_TIME	|TRAIN_TIMES	|ACCURACY 	|
|:-------------:|:-----:|:-------------:|:-----------------:|:-------------:|:---------:| 
|8				|11		|99.29			|95.69				|35.69			|0.9862		|
|16				|11		|49.66			|46.51				|12.07			|0.9865		|
|16				|9		|48.18			|46.59				|12.56			|0.9885		|
|32				|9		|36.17			|33.15				|11.81			|0.9780		|

It was found that when 'pixel_per_cell' was set to 16 and 'orient' was set to 9, then feature extracting times were half of what it initially took and the classifier accuracy was almost the same. Therefore, feature extracting parameters were chosen as follow:

|colorspace 	|pixel_per_cel 	|cell_per_block 	|orient 	|hog_channel 	|spatial_size 	|
|:-------------:|:-------------:|:-----------------:|:---------:|:-------------:|:-------------:|
|'YUV'			|16				|2					|9			|'All'			|(32,32)		|


### Sliding Window Search

#### 1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is in `window_search.py` file.

Number of sliding windows affected the performance of the vehicle detection algorithm; the more number, the slower the algorithm runs. So, the number of windows needs to be optimized to yield a faster execution time. This sliding window function was implemented such that windows sweep bottom half of an image only where cars appeared in the image. Also, different sliding window sizes were used because cars appeared in different sizes in different location of an image; smaller window scale was used near the horizon as a car appeared small near the horizon, while bigger window scale was used near the camera as a car appear big near the camera. Sample images with sliding windows were generated to visualize where those windows actually sweep through. Results are as follow:

![sliding window 1 example][image4]
![sliding window 2 example][image5]
![sliding window 3 example][image6]
![sliding window 4 example][image7]

Then, `search_windows()` function was created to collect boxes that detected car features in an image using the trained classifier above. Then, these boxes were drawn on the image using `draw_box()` function. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The code for this step is in `pipeline.py` file.

In the vehicle detection pipeline, `sliding_windows()` and `search_windows()` functions were used to extract features and to collect boxes of detected cars within a given frame using trained Linear SVC. To result better classifier performance, `sliding_windows()` and `search_windows()` function parameters were tuned.

Following images are test image examples processed by the pipeline.

![test image pipeline1 example][image8]
![test image pipeline2 example][image9]
![test image pipeline3 example][image10]
![test image pipeline4 example][image11]
![test image pipeline5 example][image12]
![test image pipeline6 example][image13]

#### 3. Describe how you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is in `pipeline.py` and `window_search.py` files.

The generated car-detected boxes were saved, and theses were used to create a heatmap. Also, by using `collections.deque`, heatmap of last few video frames were stored and they got averaged. This helped minimizing the noises in the box size change and also helped rejecting false positives.

Then, a threshold was applied on the averaged heatmap to identify correct vehicle positions and minimize false positives. This threshold value was fine-tuned to get a result with the best vehicle detection performance and minimal false positives at the same time Then, `scipy.ndimage.measurements.label()` was used to identify individual boxes in the heatmap and these were assumed to be detected vehicle positions. Lastly, `draw_labeled_boxes` function was used to construct bounding boxes around the identified individual blobs.

Here's an example result showing the heatmap from test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six test imaes and their corresponding heatmaps:

![test image heatmap1 example][image14]
![test image heatmap2 example][image15]
![test image heatmap3 example][image16]
![test image heatmap4 example][image17]
![test image heatmap5 example][image18]
![test image heatmap6 example][image19]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six test images:
![test image label1 example][image20]
![test image label2 example][image21]
![test image label3 example][image22]
![test image label4 example][image23]
![test image label5 example][image24]
![test image label6 example][image25]

### Here the resulting bounding boxes are drawn onto the test images:
![test image bounding box1 example][image26]
![test image bounding box2 example][image27]
![test image bounding box3 example][image28]
![test image bounding box4 example][image29]
![test image bounding box5 example][image30]
![test image bounding box6 example][image31]

---

### Video Implementation

#### 1. Provide a link to your final video output.
A project video was fed to the pipeline frame by frame, and it produced a vehicle detected video with few false positives.
Here's a [link to my video result](./test_videos_output/processed_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?

1. Though the vehicle detectino algorithm performed well on detecting vehicles, it was not, yet, robust enough to reject false positives. For example, sudden brightness change in an image or some patterns on paved road could generate similar features that resemble features of vehicles.

2. When vehicles are close each others, then `scipy.ndimage.measurements.label()` perceives them as a single large vehicle.

3. The process time of current vehicle detection algorithm pipeline was too slow that it took around 20 minutes to process 50 seconds long video. Though the process time depends on the quality of hardwares, like a processor or GPU, still, this tells us that current vehicle detection algorithm is not suitable for a real-time application.

#### 2. Future Improvements

1. Current vehicle detection algorithm was implemented with a simple Linear Support Vector Machine Classifier, so non-linear machine learning classifiers, like SVC or Decision Trees, could be used to improve the classifier accuracy. Also, a deep neural network with convolutional layers also can be tried to yield better classifier performance.

2. An extra algorithm that can distinguish between a large vehicle and two vehicles close by needs to be implemented.

3. Current vehicle detectino algorithm should be redesigned in order to have much faster execution time. For example, 'YOLO' algorithm by Joseph Redmon uses a single neural network and this network takes whole image, divides images into regions, and predicts bounding boxes and probabilities of each region. Find the paper about YOLO from the following link: https://arxiv.org/abs/1612.08242

