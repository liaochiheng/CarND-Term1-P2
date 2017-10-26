#**Traffic Sign Recognition** 

##Writeup report

###Here is my report of tranffic sign classification project.

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

[image1]: ./report-res/visualization.png "Visualization"
[image2]: ./report-res/grayscale.png "Grayscaling"

[TrafficSign0]: ./signs/0.jpg "Traffic Sign 0"
[TrafficSign1]: ./signs/1.jpg "Traffic Sign 1"
[TrafficSign2]: ./signs/2.jpg "Traffic Sign 2"
[TrafficSign3]: ./signs/3.jpg "Traffic Sign 3"
[TrafficSign4]: ./signs/4.jpg "Traffic Sign 4"
[TrafficSign5]: ./signs/5.jpg "Traffic Sign 5"
[TrafficSign6]: ./signs/6.jpg "Traffic Sign 6"
[TrafficSign7]: ./signs/7.jpg "Traffic Sign 7"
[TrafficSign8]: ./signs/8.jpg "Traffic Sign 8"
[TrafficSign9]: ./signs/9.jpg "Traffic Sign 9"

[Softmax0]: ./report-res/softmax0.png "Probabilities for sign #0"
[Softmax1]: ./report-res/softmax1.png "Probabilities for sign #1"
[Softmax2]: ./report-res/softmax2.png "Probabilities for sign #2"
[Softmax3]: ./report-res/softmax3.png "Probabilities for sign #3"
[Softmax4]: ./report-res/softmax4.png "Probabilities for sign #4"
[Softmax5]: ./report-res/softmax5.png "Probabilities for sign #5"
[Softmax6]: ./report-res/softmax6.png "Probabilities for sign #6"
[Softmax7]: ./report-res/softmax7.png "Probabilities for sign #7"
[Softmax8]: ./report-res/softmax8.png "Probabilities for sign #8"
[Softmax9]: ./report-res/softmax9.png "Probabilities for sign #9"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

I write down a code cell number for each code cell, in my IPython notebook. This number will be useful in the following introductions.
####1. Please check the ipynb file for my implementation code.

###Data Set Summary & Exploration

####1. Loading and analying would be found in Code-cell #1 and #2.

The code for this step is contained in the Code-cell #2 of the IPython notebook.  

I used the python build-in functions to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Exploratory visualization of the dataset.

The code for this step is contained in Code-cell #3 of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data sample distributed on 43 labels.

![alt text][image1]

###Design and Test a Model Architecture

####1. Pre-process data.

In Code-cell #6, I have a pre-process function. 

As a first step, grayscaled image because the color of sign should not be cared about.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the data will have a mean of 0, and make the training more efficient and effective.
   
####2. Split and Augment training and validation data.

I didn't split training and validation data yet.

In Code-cell #4, I have a balance function including augmenting.
   
I found the distribution on 43 labels of train-dataset is very disproportional. One label takes more than 2000 samples, and one takes less than 200 samples.
   
This function is to balance the distribution. That means it will generate some images from train images, and make some operations on image, such as: blur, noise, rotate.
   
I tried balancing, but it led to overfitting. It turned out the training accuracy is going to 99% very quickly, but the validation accuracy increases very slow, and less than 80% the most.(Maybe in this situation, number of model layers is somehow small, which would also lead to overfitting).
   
I didn't use balance right now, because with current pre-process opereation, the accuracy is just ok.
   
In the future, i will try balance again, and i trust balance will be helpful. Maybe i will re-submit again if i got a better model with a higher accuracy.


####3. LeNet Model

The code for my final model is located in Code-cell #8. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5x32	| 5x5 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32  				|
| Convolution 5x5x64	| 5x5 stride, valid padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x64    				|
| Flatten       		| outputs 1600 									|
| Fully connected		| outputs 480 									|
| Fully connected		| outputs 336 									|
| Fully connected		| outputs 43 									|
|						|												|
|						|												|
 


####4. My hyperparameters is in Code-cell #9

EPOCHS = 20
BATCH_SIZE = 128

I kept learning rate as a tf.placeholder, considering decreasing rate along with increasing EPOCH.
Sadly, decreasing rate didn't work.

####5. Solution Approach.

The code for calculating the accuracy of the model is located in Code-cell #13.

My final model results were:
* training set accuracy of 99.3%
* validation set accuracy of 94.0%
* test set accuracy of 94.7%

My approach:
* My first model is LeNet-5 with parameters same as class.
* Training result: training accuracy increasing fast up to 99%, validation accuracy less than 80%.
  Overfitting obviously.
* I tried more EPOCH, less or more BATCH_SIZE, and smaller keep_prob.
  The result didn't change: overfitting.
* I guess the disproportion of train data maybe the problem, so i balanced them to approximately the same number for each label. I generate corresponding images for each label from training data with blurring, noising or rotating.
  Sadly, overfit WORSE...
* I guess maybe the model layers are the problem. I discuss this model with my friends, and found out that the depth of my model is not deep enough.
  My current layer are like this: conv1_w = 5x5x6, conv2_w = 5x5x16. full_conn1 = 240, full_conn2 = 84.
  The 5, 16, 240, 84 is too low...
  I modified these layers like this: conv1_w = 5x5x32, conv2_w = 5x5x64. full_conn1 = 480, full_conn2 = 336.
  It turns out it works!!! Not that overfitting, but still a little overfitting.
  I decreased the keep_prob from 0.5 to 0.3, that would makes more dropout. I think this will help with decreasing overfitting.
  The result is GOOD. I got the final model with an acceptable accuracy.
* I know that the accuracy is able to be higher, and i will keep tunning the model in the future.
 

###Test a Model on New Images

####1. Test on images found on the web.

Here are ten German traffic signs that I found on the web:

![alt text][TrafficSign0] ![alt text][TrafficSign1] ![alt text][TrafficSign2] ![alt text][TrafficSign3] ![alt text][TrafficSign4]
![alt text][TrafficSign5] ![alt text][TrafficSign6] ![alt text][TrafficSign7] ![alt text][TrafficSign8] ![alt text][TrafficSign9]

Actually, i got an accuracy of 40%. But 5 of those are not included in train set, which means they should be wrong.
I think the final accuracy will be 80%(4/5) exclude those 5 signs(#0, #3, #5, #7, #8)

####2. Predictions on these new traffic signs.

The code for making predictions on my final model is located in Code-cell #16 and #17.
Calculation is in #16, and #17 visualized the traffic sign and the predicted sign in contrast.

Here are the results of the prediction:

* 0.jpg: Wrong. Not included in the train set, should be wrong.
* 1.jpg: Wrong. I guess it's the backgroud that made my model confused.
         It's colorfull, but the train dada for this sign is almost white backgrouded.
         If we see the top-k graph, it tells that my model predicted this sign the right answer(30), but not sure...
* 2.jpg: Right.
* 3.jpg: Wrong. Not included in the train set, should be wrong.
* 4.jpg: Right.
* 5.jpg: Wrong. Not included in the train set, should be wrong.
* 6.jpg: Right.
* 7.jpg: Wrong. Not included in the train set, should be wrong.
* 8.jpg: Wrong. Not included in the train set, should be wrong.
* 9.jpg: Right

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is less than the accuracy on test accuracy. I think the sample for testing-on-web is too short.

####3. Softmax probabilities.

The code for showing softmax probabilities on my final model is located in the Code-cell #20.

The softmax probabilities of those traffic signs(#1, #2, #4, #6, #9) are as follows(exclude those 5 not-included in training data):

* For the sign #1, the top five soft max probabilities were
  ![alt text][Softmax1]
  The model is confused about this image. Note it recognized the right sign(Label 30), but not sure.

* For the sign #2, the top five soft max probabilities were
  ![alt text][Softmax2]

* For the sign #4, the top five soft max probabilities were
  ![alt text][Softmax4]

* For the sign #6, the top five soft max probabilities were
  ![alt text][Softmax6]

* For the sign #9, the top five soft max probabilities were
  ![alt text][Softmax9]