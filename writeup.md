#**Traffic Sign Recognition** 

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

---
###Writeup / README

###Data Set Summary & Exploration

####1. Basic summary of the data set

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library's len and shape functions to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of test set is  12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

In this section I choose to pick and show a random traffic sign image from the dataset a matplotlib plot.


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

During preprocessing, I first tried to convert images to grayscale using cv2.cvtColor function, however, once the image was converted to grayscale I couldn't get the number of channels as 1 in gray scale image. The image shape was always 32x32, not 32x32x1 even after specifying number of channels as 1 explicitly in cv2 function. So I decided to keep working with color images. I shuffled the training data in this step.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I used the validation set data (valid.p) which was provided in the project along with train and test data. 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			        |                                               |
| Convolution 3x3		| 1x1 stride, valid padding, outputs 14x14x16	|
| RELU                  |                                               |
| Avg pooling	      	| 2x2 stride,  outputs 10x10x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32   |
| Avg pooling	      	| 2x2 stride,  outputs 5x5x32 				    |
| Fully connected		| shape=(512,170)								|
| Dropout         		| keep_prob=0.62     							|
| Fully connected		| shape=(170,115)								|
| Fully connected		| shape=(115,43)								|
| Softmax				|           									|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the seventh cell of the ipython notebook. 

To train the model, I used an 34 epochs with batch size of 128. The learning rate was set to 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the seventh cell of the Ipython notebook.

My final model results were:
* validation set accuracy of 0.961
* test set accuracy of  0.942

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
LeNet was choosen as first architecture since it provides a good starting point for image classification.
* What were some problems with the initial architecture?
THe accuracy of LeNet didn't surpass 0.89 on the validation data irrespective of adjustments made to batch size, number of epochs and hyperparameters.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. 
The architecture was expanded by adding a new convolution layer, replacing max pooling by average pooling and addition of a dropout regularizer after first fully connected layer.
* Which parameters were tuned? How were they adjusted and why?
I have experimented with the number of epochs, batch_size, learning_rate and keep_prob. The number of epochs was used to determin when to stop training; smaller batch_size with lower learning rates were tried to see if the slow learning yeilds any accuracy gain; keep_prob was fine tuned to avoid under/overfitting.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The addition of a convolution layer and introduction of droput layer after first fully connected layer improve the validation set accuracy from 0.89 to 0.961

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image1] ![alt text][image1] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

The code for making predictions on my final model is located in the ninth and tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									|
| No Passing      		| No Passing  									|
| Snow/ice possible		| Ahead only   									|
| 50 km/h	      		| 50 km/hr  					 				|
| Stop Sign      		| No Entry   									| 
| Yield					| Yield											|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 5 of the 7 traffic signs, which gives an accuracy of 71.4%. This is less then the accuracy of test set as I believe that quality of some of the images suffered when they were scaled to 32x32 size leading to poor classification.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a no entry sign (probability of 0.964), and the image does contain a no entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .964         			| No Entry   									| 
| .271     				| Traffic signals								|
| .154					| Speed limit (20km/h)							|
| .148	      			| General caution				 				|
| .125				    | Go straight or left  							|

For the second image, the model is less sure that this is a no passing sign (probability of 0.137), and the image does contain a no passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .137         			| No Passing   									| 
| .099     				| Right-of-way at the next intersection			|
| .077					| Priority road     							|
| .075	      			| Speed limit (100km/h)			 				|
| .058				    | Pedestrians       							|

For the third image, the model is unsure that this is a snow/ice possible sign (probability of 0.095). It classifies the sign incorrectly as Ahead only. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .095         			| Ahead only 									| 
| .076     				| Right-of-way at the next intersection			|
| .055					| Road work                 					|
| .049	      			| Traffic signals				 				|
| .043				    | Turn right ahead  							|

For the fourth image, the model is relatively sure that this is a 50 km/hr sign (probability of 0.858), and the image does contain a 50 km/hr sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .858        			| Speed limit (50km/h)							| 
| .349     				| Speed limit (80km/h)							|
| .282					| Speed limit (30km/h)							|
| .169	      			| Wild animals crossing			 				|
| .109				    | Speed limit (60km/h) 							|

For the fifth image, the model is reasonably sure that this is a no entry sign (probability of 0.45, next probability is low but correct sign) but the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .45         			| No Entry   									| 
| .162     				| Stop							            	|
| .089					| Speed limit (20km/h)							|
| .073	      			| Traffic signals				 				|
| .070				    | Speed limit (50km/h) 							|

For the sixth image, the model is reasonably sure that this is a yeild sign (probability of 0.407, next probability is very low), and the image does contain a yeild sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .407         			| Yield   				    					| 
| .068     				| Speed limit (20km/h)  						|
| .068					| Bicycles crossing  							|
| .058	      			| Speed limit (30km/h)			 				|
| .054				    | Ahead only        							|

For the seventh image, the model is not quite sure that this is a slippery road sign (probability of 0.201), the image does contain a slippery road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .201         			| Slippery road									| 
| .19     				| Pedestrians   								|
| .18					| Dangerous curve to the left   				|
| .177	      			| Right-of-way at the next intersection			|
| .09				    | Dangerous curve to the right   				|

 