#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

###Data Set Summary & Exploration

###1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python built-in funciton 'len()' to get the number of each data sets:

* The size of training set: 34799
* The size of the validation set: 4410
* The size of test set: 12630
* The shape of a traffic sign image: (32,32,3)
* The number of unique classes/labels in the data set: 43

I also included assert(len(feature) == len(label)) to ensure number of features and number of labels are the same in each dataset.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first bar chart shows the number of images by each dataset; The whole dataset is distributed to about 70% of training set, 10% of validation set, and 20% of test set. The dataset split is reasonable as the total number of image is ~52000 and this is considered relatively small dataset.


![alt text][image1]

Second bar chart shows the number of training images by each class/label. The chart shows that number of training images of some classes are small compared to other classes. Number of training images can be increased by introducing the data augmentation techniques.

###Design and Test a Model Architecture

###1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I decided not to grayscale dataset, but only normalize them. Compared to the MNIST data, this traffic sign dataset can be distinguished by not only its shape and pattern, but also its color. So, grayscaling dataset could lose significant information that can be trained by a model.

I used Min-Max Scaling to normalize dataset. This normalizing input helps to make the cost function more symmetric in space and this makes an optimizer to optimize the cost function more easily with higher learning rate.

For the data augmentation, this is something that needs to be carefully processed. In this project context, data augmentation can mean the image distortion, like image rotation, change in color, introducing noises, etc. But, image rotation and change in color can introduce error in the training set. For example, if an ‘Ahead Only’ sign is rotated, then this can be looked the same with a ‘Keep Right’ or ‘Keep Left’ sign. Thus, in this project context, traffic sign images should not be rotated or changed in color, but they can only be distorted in such ways, like changing brightness, making blurry, etc.


###2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consists of following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Normalized image   					| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x40 	|
| Max pooling	      	| 2x2 stride, SAME padding, outputs 14x14x40	|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x80 	|
| Max pooling	      	| 2x2 stride, SAME padding, outputs 5x5x80	 	|
| Flattened layer      	| outputs 1x2000							 	|
| Fully connected		| Dropout 0.75, output 1x1000					|
| Fully connected		| Dropout 0.75, output 1x300					|
| Fully connected		| Dropout 0.75, output 1x100					|
| Output layer			| output 1x43									|
 
I used Leaky ReLU activation with alpha = 0.15 instaed of normal ReLU on every convolutional layer and fully connected layer because all negative input values to layers are discarded by normal ReLU and this could lose some significant information.

I also implemented Batch-Normalization in every layer. Like input normalization, batch-norm normalize the value by shifting the mean and by adjusting variance before they are fed into an activation function within every layer. This helps to train a model faster and yield a model with better performance.

###3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer, which is an adaptive moment estimation and it combines RMS prop and Momentum techniques. I chose 512 Batch Size and 50 Epochs that are relatively higher because the final model is large and it needs to be trained longer. Learning rate value was set to 0.0005 and this was set after few trials of training.

###4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 100.00%
* validation set accuracy of 98.32%
* test set accuracy of 97.82%

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

First, I tried out with simple LeNet structure: 2 convolution layers and 2 fully connected layers. This is simple and well performing image classifying model, so it was a good starting point.

* What were some problems with the initial architecture?

The problem with the simple LeNet architecture was its validation accuracy was too low; it was around 85%. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Since the accuracy of the first architecture was too low and this indicates the underfitting of the model, it was thought that first architecture was small model for 43 classes. So I increased the model architecture size: Number of output in each convolution layer was increased, number of output of each fully connected layer was increased, and one more fully connected layer was added to the model. And, with some hyperparameter tuning, the performance of the model increased higher than 90%.

However, a gap between training error and validation error was quite big, like around 10%,  and this indicates the overfitting of the model to the training data. This variance problem can be solved by regularizations. So, I added dropout to fully connected layers and implemented batch-normalization to convolution layers and fully connected layerrs. With some hyperparamter tuning, again, the validation error reduced down close to the training error.

* Which parameters were tuned? How were they adjusted and why?

Learning rate was tuned specifically as it is the hyperparameter that is directly related to the cost function, so it heavily affects the performance of the model training. I started with low value around 0.001 and tried training the model with more values by increasing/decreasing by 50%.

Batch size was tuned between 128 and 1024, and Epoch was tuned between 10 and 80. I repeated tuning them until the model yields better accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

###1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

These are six German traffic signs that I downloaded from 'benchmakr.ini.rub.de' web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The second image could be difficult to classify as there is a noise in the image. And, the fourth and sixth images could be difficult to classify because the brightness is too low and they are even quite difficult for a human to classify.

###2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| End of no passing ... | End of no passing ... 						|
| Bicycles crossing		| Bicycles crossing								|
| No Passing      		| No Passing 					 				|
| Go straight or right	| Go straight or right							|
| Speed limit (120 km/h)| Speed limit (120 km/h)						|


The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. 

###3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all images, the model is 100% sure about its predictions. 


However, since only 6 images are not enough to correctly evaluate the trained model, test dataset was used to get the test accuracy of the trained model. The final model yielded 97.82% of test accuracy and I randomly chose and displayed 6 images with their softmax probability results.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


