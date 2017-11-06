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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

First, I tried with 5 convolutional layers and 3 fully connected layers as it was suggested in the lecture. However, after doing some research, I found that a smaller architecture can train the model well, so I reduced the model to 3 convolutional layers and 2 fully connected layers.

To see how well the model was working, I trained model with original collected data without any augmentation. I split the dataset to training and validation set to check if the model is trained properly.

Then, I tried to tune hyperparameters, like learning rate and number of epochs, first, by observing the change in losses. I selected values that yielded small starting loss value and smooth decrease in both losses.

Also, significant difference between training loss and validation loss, which indicates the model overfitting, so I added MaxPooling layer and Batch Normalization layer. 

#### 2. Final Model Architecture



![alt text][image1]

#### 3. Creation of the Training Set & Training Process

I started training process with Udacity dataset. Then, I collected more data of two laps in both directions and another lap focusing on curve lanes. Before the collected dataset is used to train a model, histogram was made to see the distribution of steering angle.

![alt text][image2]

Since the track is mostly straight track, original data is highly biased with dataset with steering angle close to 0. So, this dataset was processed to drop off about 80% of data with steering angle less than 0.05 deg.

After training the model with these data, I deployed the model in the simulation, and I found that the vehicle was not able to recover from deviating from the track. So, I collected more data of recovery driving from left side and right side to the center of the track.

After collecting data, I had about 23K number of data points (all three camera images). Before I use them for training, I used data agumentation to create even more data. I flipped images if its steering angle is greater than 0.3 deg, changed the brightness of image, horizontally shifted image, blurred image, and created random shadow. These data augmentation was randomly applied to an image by rolling a dice.

Also after images wre augmented, all images were cropped (55 pixels from top and 15 pixels from bottom), normalized, and resized to 64x64 image.

Lastly, these data were randomly shuffled and were splited into 70% of training set and 30% of validation set. The validation set was used to indicate whether the model was overfitting or underfitting.

Epoch was chosen to 8 because of model was trained more than 10 epochs, then the vehicle was not able stay in the line.
