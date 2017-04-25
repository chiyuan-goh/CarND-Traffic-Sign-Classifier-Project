#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[classd]: imgs/class_distribution.png "Class Distribution"
[image1]: imgs/23_slippery.jpg  "sign 1"
[image2]: imgs/17_noentry.jpg  "sign 2"
[image3]: imgs/25_roadwork.jpg  "sign 3"
[image4]: imgs/22_bumpy.jpg  "sign 4"
[image5]: imgs/01_speed30.jpg  "sign 5"
[image6]: imgs/validation.png "valid"
[image7]: imgs/normalized.jpg "norm"


## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32 by 32 by 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the class distribution of the dataset. Overall, we can observe that the class distributions are not evenly distributed, with low numbers in classes such as "20km/h Speed limit" (Class 0) and high in e.g Yield (class 13). This is likely due to the availability of such signs in real life i.e there are very little areas in Germany where the speed limit is indeed 20km/h. However, this distribution is fairly consistent across the test, validation and training dataset, hence it is likely that they are sampled in a stratified manner.

![alt text][classd]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, we decided to convert the images to grayscale. This decision is emprical; for the same baseline model, a slightly better performance ~1.5%. Since the number of weights to learn is smaller for grayscale inputs, we have stuck to this.

Here is an example of a traffic sign image before and after grayscaling/noralization.

![alt text][image7]

As a last step, I normalized the image data, subtracting each pixel by an approximate mean of 128. and dividing by approximate standard deviation of 128, in order to control the input values to similar ranges so that gradients wouldn't be dominated by overly large values during backpropagation training.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x38 	|
| Activation					| Relu												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 14x14x38 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x64      									|
| Activation | Relu |
| Max pooling 2x2 | 2x2 strides, VALID padding, outputs 5x5x64
| Fully connected		| 120 output nodes        									|
| Activation | Relu
| Fully connected				| 84 output nodes        									|
| Activation | Relu
| output						| 43 nodes == number of classes												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Tensorflow's tf.nn.AdamOptmizer which is a variant of SGD that adjusts the learning rate. Batch size is fixed untuned at 128, and the learning rate is set to 0.001. There were some [pro-tips](http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/) that advises applying dropout and increasing learning rate by a large factor, but it was observed that the validation and training accuracy swing wildly between iterations. Number of epochs started at 10 and seemingly converge around ~60 for the hyperparameters and architectures experimented. We did not observe overfitting from 10 - 60.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

- training set accuracy of 100%
- validation set accuracy of 96.8% 
- test set accuracy of 94.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Yann Lecun's LeNet5 archicture for the Minst dataset was chosen as the baseline model. The reason for this is that the small number of convolution/fully connected layers and simple layout allows us to "add" up, make adjustments and observe the changes in performance.

![alt text][image6]

The validation curve over epoches on the baseline above does not seem to provide strong evidence that the model is either overfitting or underfitting. True enough, adding dropout to the connected layers does not yield better accuracy, albeit longer convergence. In the early stages of learning, we tried adding extra connected layers to the end of network successfully. Finally, I took inspiration from the paper [Traffic Sign Recognition with Multiscale Convolution Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) on the direction to take. Particularly, I learnt the lesson of matching architecture of CNN to traditional hand-crafted feature classification approach, in that:

* convolution layers <-> handcrafted features
* fully connected layers <-> classifier

It is then straight-forward that more attention should be paid on the former. Getting more, better features are always good. In the event that the number of features are too high that the model overfits, I can increase the number of training samples through data augmentation.

Finally, I scaled the depth of the filter up from 6 to 84 in the first layer, and 16 to 64 in the second. This increases the validation accuracy by ~3.5%.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. These are chosen on the basis of testing the limits of the network i.e they exhibit certain qualities that the training data do not possess:

This is a bumpy road sign, but heavily vandalised. Even a human will have trouble distinguishing the true label of this sign. 

![alt text][image4]



This is a 30km/h speed limit sign. It is partially occluded and rotated.

![alt text][image5]


This is a road works sign. It is partially occluded, and the orientation of the sign is not parallel to the image plane.
 ![alt text][image3] 


This is a bumpy road sign. It is heavily scaled up and occluded.
 ![alt text][image1]


This is a No Entry sign. The words "Do Not Enter" are added to the sign.
![alt text][image2]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Bumpy Road    			| General Caution										|
| Slippery Road					| 30 km/h Limit											|
| 30 km/h Limit	      		| 50 km/h Limit					 				|
| Road Works		| Keep Right      							|


The model was able to correctly predict 1/5 samples, which gives an accuracy of 20.0%. This does not compares favorably to the accuracy on the test set of 94.7%. While we identify these as edge cases, they are in fact fairly common in reality. Humans would have no problems recognizing these "distorted" signs, hence to achieve high level of autonomy, the network should be able to distinguish these.

It is interesting that for the No Entry sample, the network is ~100% confident of its prediction despite its distortion. Even more notable is that even though the predictions are incorrect for Bumpy Road, 30km/h and Road Works, the network is equally confident. This seemingly suggested that the network is intolerable to scale, perspective rotational transformation.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell #TODO of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

True Label          	|     P1 	        					|  P2 | P3 | P4 | P5
|:----:|:---------------------:|:---------------------------------------------:| :---:| :---:|:-----:|
| No Entry         			| No Entry (~1.0) | Yield (~0.) | Wild Animals (~0.) | Traffic Signal (~0.) | Stop (~0.)    									| 
| Bumpy Road     				| General caution (0.91) | Traffic Signals (~0.09) | Road work (~0.) | Go straight/Left (~0.) | Speed 30km/h limit (~0.)  										|
| Slippery Road					| 30Km/h (~1.0) | General Caution (~0.) | 20Km/h (~0.0) | 50Km/h (~0.0) | Road Work (~0.0)											|
| 30Km/h	      			| 50Km/h (~1.0) | Keep left (~0.) | Yield (~0.) | 30Km/h (~0.) | 60Km/h (~0.) 					 				|
| Road Work				    | Keep Right (0.43) | Stop (0.42) | Turn Left Ahead (0.14) | End of limits (~0.) | Children Crossing (~0.)      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The first layer feature map is easy to interpret, and largely attest to edges and lines in the images. The second layer feature map is harder to distinguish, so I manually created a custom image to look out for high activations in the channel. For instance, for channel 24 (see ipython notebook), comparing with activations from the previous images, intuitively we can hypothesize that it correspond to squarish/rectangular regions of smooth gradients.  


