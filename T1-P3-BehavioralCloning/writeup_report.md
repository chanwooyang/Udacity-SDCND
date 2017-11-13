#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

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

Lastly, these data were randomly shuffled and were splited into 70% of training set and 30% of validation set. The validation set was used to indicate whether the model was overfitting or underfitting.