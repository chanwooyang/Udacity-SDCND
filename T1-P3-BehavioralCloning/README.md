# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

-------


# Behavioral Cloning Project: Writeup Report

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_img/distribution_raw.png "Raw Data Distribution"
[image2]: ./writeup_img/distribution_down.png "Downsampled Data Distribution"
[image3]: ./writeup_img/distribution_down_trim.png "Downsampled+Trimmed Data Distribution"
[image4]: ./writeup_img/image.png "Original Image as a Reference"
[image5]: ./writeup_img/flipImage.png "Flipped Image"
[image6]: ./writeup_img/brightImage.png "Bright Image"
[image7]: ./writeup_img/transImage.png "Translated Image"
[image8]: ./writeup_img/shadowImage.png "Shadow Image"
[image9]: ./writeup_img/blurImage.png "Blur Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* dataset_generator.py containing the script to collect, preprocess, and generate dataset
* data_visualize.py containing the script to show histogram of dataset and sample preprocessed images
* drive.py for driving the car in autonomous mode
* model_4.h5 containing a trained neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_4.h5
```

Using data_visualize.py, distribution of steering angles of current dataset and sample augmented images can be shown.
```sh
python data_visualize.py
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows model architecture and how I splitted training and validating sets for the model.

### Model Architecture

#### 1. Model Architecture Design

My model is based on NVIDIA model: 5 convolutional layers and 3 fully connected layers. Normalizing and cropping layers are added prior to convolutional layers. ELU activation function was used to introduce a non-linearity in the model.

#### 2. Attempts to reduce overfitting in the model

Dropout layers with keep probability of 0.2 were added after each of first two fully connected layers.

#### 3. Model parameter tuning

Since Keras Adam Optimizer was used, the learning rate was not manually tuned. 

Batch Size was tried with 128, 256, and 512. Model was trained with small set of data with different Batch Size to see to what degree it affects the model training. There was not a significant difference, but Batch Size of 128 yielded the best results, so 128 was chosen.

For Epoch, 'model.compile()', 'model.fit_generator()', and 'model.save()' were put in a for loop so that a trained model gets saved separately every Epoch. By testing each epoch-trained model, model overfitting due to too long training time can be prevented and the best performing model can be easily chosen. Epoch for for-loop was set to 8 because it was observed that a model tends to get overfitted after 5 Epochs.

Any other model parameters, like kernel size, depth of layer, etc., remained the same with the NVIDIA model.

#### 4. Final Model Architecture

Final model architecture looks the following:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB Image  	 						| 
| Normalize 	    	| pixel/255.0 - 0.5, outputs 160x320x3 			|
| Crop 	    			| Crop Top 55 Bot 25 pixels, outputs 80x320x3 	|
| Convolution 1x1     	| 1x1 stride, SAME padding, outputs 80X320X1 	|
| Convolution 5x5	    | 2X2 stride, VALID padding, output depth: 24 	|
| Convolution 5x5	    | 2X2 stride, VALID padding, output depth: 36 	|
| Convolution 3x3	    | 2X2 stride, VALID padding, output depth: 48 	|
| Convolution 3x3	    | 1x1 stride, VALID padding, output depth: 64 	|
| Convolution 3x3	    | 1x1 stride, VALID padding, output depth: 64 	|
| Flattened layer      	| Flatten output of convolutional layer		 	|
| Fully connected		| Dropout 0.2, output depth: 100				|
| Fully connected		| Dropout 0.2, output depth: 50					|
| Fully connected		| Output depth: 10								|
| Output layer			| output 1x1									|


### Training Strategy

#### 1. Data Collection Strategy

Starting with given Udacity dataset, more data was collected from the Udacity simulator; two laps in both Clock-Wise direction and Counter Clock Wise direction were recorded. A model was trained with these data, and was tested on the track. During the test run, it was observed that the vehicle went off track at a curved track and at the bridge, so more driving data at those spots (specifically curves and bride) were recorded. Left and Right Camera images were used if the steering angle was greater than 0.05 deg. For these images, correction factor of 0.25 deg was added and subtracted to a steering angle for Left and Right Camera images, repectively. 
These collected data was downsampled and augmented before they were used for the training. Details are explained below.

#### 2. Downsampling Dataset

With collected data, histogram over steering angles was made to evaluate the distribution or 'balance' of collected data. The following histogram shows the distribution of collected raw data:

![Raw Data Distribution][image1]

As it is shown, the data is highly biased at angle near 0 deg and near +/-0.25 deg (this is due to the correction factor for Left and Right Camera images). This could introduce a significant noise in training process, so if the steering angle is smaller than 0.05 deg, then they were discarded at 60% of probability.

![Downsampled Data Distribution][image2]

Though it is downsampled, the distribution is still bit biased at some angle ranges, so further data trimming has been done.

![Downsampled+Trimmed Data Distribution][image3]

Now, the data distribution looks bit more smooth and similar to the normal distribution.

#### 3. Data Augmentation

To increase the data size for training, it is possible to collect even more data by running the Udacity Simulator, but it can be increased by different data augmentations. Followings are types of data augmentation used in this project:

This is an original image as a reference to augmented images:

![Original Image][image4]

##### 1. Flip Image
If absolute value of steering angle is greater than 0.3 deg, then the image was flipped horizontally and the steering angle was multiplied by -1

![Flipped Image][image5]

##### 2. Brightness Change
A brightness of an image was changed using 'Gamma Correction'

![Bright Image][image6]

##### 3. Horizontal Translation
An image was horizontally translated and its steering angle was corrected based on how much pixels it was translaed

![Bright Image][image7]

##### 4. Insert Shadow
A random shadow was created on to an image

![Bright Image][image8]

##### 5. Blur Image
An image was slightly blurred using cv2.GaussianBlur() function

![Bright Image][image9]

All these data augmentations were randomly applied to an image within the dataGenerator() function. Detail codes for each data augmentation can be found in 'dataset_generator.py'.


#### 4. Creation of the Training Set & Training Process

In this project, 'generator' was used to prevent the memory issue. Since there are few hundred thousands image data for training, it is almost impossible to store all dataset in memory. So rather, by using Python 'generator', 'Batch Size'-sized dataset can be generated on the fly. Within the 'generator', an image is loaded and data augmentation is randomly applied by rolling a dice.

Lastly, these data were randomly shuffled and were splited into 80% of training set and 30% of validation set. The validation set was used to indicate whether the model was overfitting or underfitting.

#### 5. Training Model

Trained model was saved every Epoch. Training was done for 8 Epoch and each of saved model was tested in the simulator. It was found that after 3 Epoch, all trained model performed well in track1, but the trained model at Epoch 5 (model_4.h5) was the best performing model. Test videos were created at different speed: [10 mph](https://github.com/chanwooyang/Udacity-SDCND-T1/blob/master/T1-P3-BehavioralCloning/Test%20Run%20Video/10mph.mp4), [20 mph](https://github.com/chanwooyang/Udacity-SDCND-T1/blob/master/T1-P3-BehavioralCloning/Test%20Run%20Video/20mph.mp4), and [30 mph](https://github.com/chanwooyang/Udacity-SDCND-T1/blob/master/T1-P3-BehavioralCloning/Test%20Run%20Video/30mph.mp4).


#### 6. Problem Discussion & Future Improvement

1. The method of using three camera and corresponding angle correction factors addressed the problem of distributional 'drift'. This also can be addressed by DAgger (Dataset Aggregation) method:
	- Train a policy (or model) from a human dataset
	- Run a policy (or model) to generate dataset
	- Ask a human to label the generated dataset with actions (steering angle in this case)
	- Aggregate the generated dataset to the human dataset
	- Repeat until a performance is improved

2. It might fail to fit/train a policy/model to an expert because of two things: 1. Non-Markovian behavior, 2. Multimodal behavior.

3. Non-Markovian behavior means that the behavior depends on all past behavior, but not only on current observation. In Markovian behavior, if a policy observe the same thing twice, then the policy do the same thing twice, regardless of what happened before, but this is very unnatural for human demonstrators. Many of human behaviors really depends on past states and observation. This problem could be resolved by introducing RNN (Recurrent Neural Network) structure in a model (typically, LSTM cells work better). 

4. Multimodal behavior is having a bad behavior distribution due to an inconsistency of a human.
Humans are quite inconsistent and they do not do the same thing twice not only because of the history, but also because of arbitrary reasons. For example, if a car avoids an obstalce, a human driver could turn left or turn right. If a policy of the car has an discrete action, it is fine because it will be like, high probability on 'turn left', 'turn right', and low probability on 'go forward'. However, if it is a continuous action or very high dimensional discrete action (smooth enough) and the resultant distribution would be more like an average of turn-left and turn-right distributions. This issue could be addressed by following techniques: 1. Output mixture of Gaussians, 2. Implicit Density Model, 3. Autoregressive Discretization


Discussions and techniques above are from [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_2_behavior_cloning.pdf) at UC Berkeley




