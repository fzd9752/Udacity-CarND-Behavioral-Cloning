# **Behavioral Cloning**

## Project Report by Elsa Wang

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.jpg "center"
[image3]: ./examples/center.jpg "Recovery Image"
[image4]: ./examples/right.jpg "Recovery Image"
[image5]: ./examples/left.jpg "Recovery Image"
[image6]: ./examples/loss.png "Loss Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing trained convolution neural networks
* writeup_report.md summarizing the results

also, the project includes:
 * run.mp4 containing a autonomous driving video with _model.py_ and _model.h5_
 * model_personal_data.h5 containing trained convolution neural networks with my personal driving data

---


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I built my model inside function _buildmodel()_ (model.py lines 111-127). It consists of 3 convolution neural network layers and 1 fully connected layers. The architecture could see in following image.

![Model architecture][image1]

Each CNN layer includes _RELU_ layers to introduce nonlinearity, and the data is normalised in the model using a Keras lambda layer (code line 113).

#### 2. Attempts to reduce overfitting in the model

The model contains _dropout_ and _Max Pooling_ layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 79 and line 100-101). The model was tested by separated data (code line 85-97), it was also testing by running it through the simulator and ensuring that the vehicle could stay on the track (video_img.mp4).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 125).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the filtered sample data provide by Udacity. Also, I also recorded my personal driving data to test the model.
For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use minimised model to guide the simulator car to drive properly.

My first step was to use a regression model. I thought this model might be appropriate because it was simple and its outputs were continuous number similar to the steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set, and a test set. I found that my first model had a extreme low mean squared error on the training set and on the validation set. However, the vehicle could only stay on the straight road. This implied that the regression model could not tell the curve road and straight road.

To improve the classifying capability, I modified the model so that the convolutional neural network layers were introduced.

I modified my model based on previous project - traffic sign classifier - which was also a model to recognise the image. I tried to add only one CNN or fc layer each time, and then tested model performance on the simulator. I've tuned the filters and depths several times to make the model have more powerful to recognise the curves and broken around track one.

With only 90,000+ parameters, the vehicle is able to drive autonomously around the track without leaving the road with a middle speed(~15-22mph).

#### 2. Final Model Architecture

The final model architecture (model.py lines 111-127) consisted of 3 convolution neural network layers and 1 fully-connected layers.

Here is a visualisation of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I chosen the sample data that provided by Udacity. Here is an example image of centre lane driving:

![alt text][image2]

I randomly filtered a few of 0 steering images, because they caused unbalanced of the steering data and would disturb the model training accuracy. After that, I had 4969 number of images. I split 10% into a test set. The left were as the training set.

To augment the data set, I used images from left and right cameras as recovering data so that the vehicle would learn to back to the road centre. These images show what a recovery looks like starting from too left and too right of the road:

![alt text][image4]
![alt text][image5]

I also flipped images and angles thinking that this would balance the data to avoid too much turning left.

After the collection process, I had 26832 number of data points. I then preprocessed this data by resizing and cropping road irrelevant parts.

I finally randomly shuffled the data set and put 20% of the training data into a validation set.

I used this training data for training the model. The validation set and test set helped determine if the model was over or under fitting.

I used an adam optimizer so that manually training the learning rate wasn't necessary.

### Test of model architecture and future

In order to test the cloning ability of the model architecture, I recorded two tracks driving by myself. Although there was **not** any extra recovery data, the vehicle can also keep on the centre road at the most of time with the moderate speed (try model_personal_data.h5, also it clone my bad behaviour of out right curve).

My personal goal of this project is to deep understanding CNN and mastering my skills. It is an exciting process to only use 90k+ params and my MacBook to build a model to drive a car. I did not grayscale inputs, which seems to downsize the data and increase the accuracy. I will  challenge the track 2 in the future.
