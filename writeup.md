# **Traffic Sign Recognition Using Deep Learning** 



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


Summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is 32 by 32 by 3
* The number of unique classes/labels in the data set is 43


Here is an exploratory visualization of the data set. It is a bar chart showing the class distribution of the dataset. Overall, we can observe that the class distributions are not evenly distributed, with low numbers in classes such as "20km/h Speed limit" (Class 0) and high in e.g Yield (class 13). This is likely due to the availability of such signs in real life i.e there are very little areas in Germany where the speed limit is indeed 20km/h. However, this distribution is fairly consistent across the test, validation and training dataset, hence it is likely that they are sampled in a stratified manner.

![alt text][classd]

### Experiment

#### Image Preprocessing

As a first step, I decided to convert the images to grayscale. This decision is based on empirical envidence; for the same baseline model, I have observed a slightly better accuracy of approximately ~1.5%. Since the number of weights to learn is smaller for grayscale inputs, I have stuck to this.

As a last step, I normalized the image data, subtracting each pixel by an approximate mean of 128. and dividing by approximate standard deviation of 128, in order to control the input values to similar ranges so that gradients wouldn't be dominated by overly large values during backpropagation training.

Here is an example of a traffic sign image before and after grayscaling/noralization.

![alt text][image7]


#### Architecture

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
 



To train the model, I used Tensorflow's tf.nn.AdamOptmizer which is a variant of SGD that adjusts the learning rate. Batch size is fixed untuned at 128, and the learning rate is set to 0.001. There were some [pro-tips](http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/) that advises applying dropout and increasing learning rate by 10x, but it was observed that the validation and training accuracies swing wildly between iterations. Number of epochs started at 10 and seemingly converge around ~60 for the hyperparameters and architectures experimented. I did not notice overfitting when increasing the number of epochs.


My final model results were:

- training set accuracy of 100%
- validation set accuracy of 96.8% 
- test set accuracy of 94.7%

They can found on cell 8 and 17 of the notebook respectively.

#### Architecture Selection

Yann Lecun's LeNet5 archicture for the Minst dataset was chosen as the baseline model. The reason for this is that the small number of convolution/fully connected layers and simple layout allows us to "add" up, make adjustments and observe the changes in performance.

![alt text][image6]

The validation curve over epoches on the baseline above does not seem to provide strong evidence that the model is either overfitting or underfitting. True enough, adding dropout to the connected layers does not yield better accuracy, albeit longer convergence. In the early stages of learning, I tried adding extra connected layers to the end of network unsuccessfully. Finally, I took inspiration from the paper [Traffic Sign Recognition with Multiscale Convolution Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) on the direction to take. Because the discussion focused on the same dataset, it is worth picking ideas off from it. Particularly, I learnt the lesson of matching architecture of CNN to traditional hand-crafted feature classification approach, in that:

* convolution layers <-> handcrafted features
* fully connected layers <-> classifier

It is then straight-forward that more attention should be paid on the former. Getting more, better features are always good. In the event that the number of features are too high that the model overfits, I can increase the number of training samples through data augmentation.

For the final model selection, I scaled the depth of the filter up from 6 to 84 in the first layer, and 16 to 64 in the second. Compared to baseline, this increases the validation accuracy by ~3.5%.

### Test a Model on New Images

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


True Label          	|     P1 	        					|  P2 | P3 | P4 | P5
|:----:|:---------------------:|:---------------------------------------------:| :---:| :---:|:-----:|
| No Entry         			| No Entry (~1.0) | Yield (~0.) | Wild Animals (~0.) | Traffic Signal (~0.) | Stop (~0.)    									| 
| Bumpy Road     				| General caution (0.91) | Traffic Signals (~0.09) | Road work (~0.) | Go straight/Left (~0.) | Speed 30km/h limit (~0.)  										|
| Slippery Road					| 30Km/h (~1.0) | General Caution (~0.) | 20Km/h (~0.0) | 50Km/h (~0.0) | Road Work (~0.0)											|
| 30Km/h	      			| 50Km/h (~1.0) | Keep left (~0.) | Yield (~0.) | 30Km/h (~0.) | 60Km/h (~0.) 					 				|
| Road Work				    | Keep Right (0.43) | Stop (0.42) | Turn Left Ahead (0.14) | End of limits (~0.) | Children Crossing (~0.)      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

The feature map visualizations can be found on cell 20 onwards in the ipython notebook.

The first layer feature map is easy to interpret, and largely attest to edges and lines in the images. The second layer feature map is harder to distinguish, so I manually created a custom image to look out for high activations in the channel. For instance, for channel 24 (see ipython notebook), comparing with activations from the previous images, intuitively we can hypothesize that it correspond to squarish/rectangular regions of smooth gradients.  


