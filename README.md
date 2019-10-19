# Image-Detection
Image Processing and Image Identification
Abstract
         This code pattern demonstrates how images, identify the images like Cat, Dog, tree, Bike, car can be classified using Convolutional Neural Network (CNN). Even though there are code patterns for image classification, none of them showcase how to use CNN to classify images using Keras libraries.
In existing system there are many techniques which are available for extracting information from images but there are no exact processing is defined. In proposed system we will come across different new techniques in image processing.
Image Recognition (Classification)
Image recognition refers to the task of inputting an image into a neural network and having it output some kind of label for that image. The label that the network outputs will correspond to a pre-defined class. There can be multiple classes that the image can be labeled as, or just one. If there is a single class, the term "recognition" is often applied, whereas a multi-class recognition task is often called "classification".
A subset of image classification is object detection, where specific instances of objects are identified as belonging to a certain class like animals, cars, or people.
Feature Extraction
In order to carry out image recognition/classification, the neural network must carry out feature extraction. Features are the elements of the data that you care about which will be fed through the network. In the specific case of image recognition, the features are the groups of pixels, like edges and points, of an object that the network will analyze for patterns.
Feature recognition (or feature extraction) is the process of pulling the relevant features out from an input image so that these features can be analyzed. Many images contain annotations or metadata about the image that helps the network find the relevant features.
How Neural Networks Learn to Recognize Images
Getting an intuition of how a neural network recognizes images will help you when you are implementing a neural network model, so let's briefly explore the image recognition process in the next few sections.
What is CNN and why CNN?
A CNN is a supervised learning technique which needs both input data and target output data to be supplied. These are classified by using their labels in order to provide a learned model for future data analysis.
Typically a CNN has three main constituents - a Convolutional Layer, a Pooling Layer and a Fully connected Dense Network. The Convolutional layer takes the input image and applies m number of nxn filters to receive a feature map. The feature map is next fed into the max pool layer which is essentially used for dimensionality reduction, it picks only the best features from the feature map. Finally, all the features are flattened and sent as input to the fully connected dense neural network which learns the weights using back propagation and provides the classification output.
The motivation behind the CNN is that it is based on the way the visual cortex functions, where one object in the scene is in focus while the rest is blurred, similarly the CNN takes one section/window of the input image at a time for classification. Each time the CNN will produce a feature map for each section, in the convolutional layer. In the Pooling layer it removes the excess features and takes only the most important features for that section, thereby performing feature extraction. Hence, with the use of CNNs we don't have to perform an additional feature extraction technique.
CNNs require lesser pre-processing as compared to other similar classification algorithms. While traditional MLP(Multi Layer Perception) algorithms have significant accuracy for image recognition, they suffer from the curse of dimensionality due to the nodes being fully connected, and hence cannot be scaled to high resolution images. CNNs overcome these challenges posed by MLP by exploiting the spatial correlation of an image. This is done by enforcing a pattern of local connectivity between adjacent neuron layers. 
Feature Extraction with Filters:
The first layer of a neural network takes in all the pixels within an image. After all the data has been fed into the network, different filters are applied to the image, which forms representations of different parts of the image. This is feature extraction and it creates "feature maps".
This process of extracting features from an image is accomplished with a "convolutional layer", and convolution is simply forming a representation of part of an image. It is from this convolution concept that we get the term Convolutional Neural Network (CNN), the type of neural network most commonly used in image classification/recognition.
If you want to visualize how creating feature maps works, think about shining a flashlight over a picture in a dark room. As you slide the beam over the picture you are learning about features of the image. A filter is what the network uses to form a representation of the image, and in this metaphor, the light from the flashlight is the filter.
The width of your flashlight's beam controls how much of the image you examine at one time, and neural networks have a similar parameter, the filter size. Filter size affects how much of the image, how many pixels, are being examined at one time. A common filter size used in CNNs is 3, and this covers both height and width, so the filter examines a 3 x 3 area of pixels.
Convolutional layer with Feature Layers:
While the filter size covers the height and width of the filter, the filter's depth must also be specified.
How does a 2D image have depth?
Digital images are rendered as height, width, and some RGB value that defines the pixel's colors, so the "depth" that is being tracked is the number of color channels the image has. Grayscale (non-color) images only have 1 color channel while color images have 3 depth channels.
All of this means that for a filter of size 3 applied to a full-color image, the dimensions of that filter will be 3 x 3 x 3. For every pixel covered by that filter, the network multiplies the filter values with the values in the pixels themselves to get a numerical representation of that pixel. This process is then done for the entire image to achieve a complete representation. The filter is moved across the rest of the image according to a parameter called "stride", which defines how many pixels the filter is to be moved by after it calculates the value in its current position. A conventional stride size for a CNN is 2.
The end result of all this calculation is a feature map. This process is typically done with more than one filter, which helps preserve the complexity of the image.
Activation Functions
After the feature map of the image has been created, the values that represent the image are passed through an activation function or activation layer. The activation function takes values that represent the image, which are in a linear form (i.e. just a list of numbers) thanks to the convolutional layer, and increases their non-linearity since images themselves are non-linear.
The typical activation function used to accomplish this is a Rectified Linear Unit (ReLU), although there are some other activation functions that are occasionally used .
Pooling Layers
After the data is activated, it is sent through a pooling layer. Pooling "downsamples" an image, meaning that it takes the information which represents the image and compresses it, making it smaller. The pooling process makes the network more flexible and more adept at recognizing objects/images based on the relevant features.
When we look at an image, we typically aren't concerned with all the information in the background of the image, only the features we care about, such as people or animals.
Similarly, a pooling layer in a CNN will abstract away the unnecessary parts of the image, keeping only the parts of the image it thinks are relevant, as controlled by the specified size of the pooling layer.
Because it has to make decisions about the most relevant parts of the image, the hope is that the network will learn only the parts of the image that truly represent the object in question. This helps prevent overfitting, where the network learns aspects of the training case too well and fails to generalize to new data.
There are various ways to pool values, but max pooling is most commonly used. Max pooling obtains the maximum value of the pixels within a single filter (within a single spot in the image). This drops 3/4ths of information, assuming 2 x 2 filters are being used.
The maximum values of the pixels are used in order to account for possible image distortions, and the parameters/size of the image are reduced in order to control for overfitting. There are other pooling types such as average pooling or sum pooling, but these aren't used as frequently because max pooling tends to yield better accuracy.
Flattening
The final layers of our CNN, the densely connected layers, require that the data is in the form of a vector to be processed. For this reason, the data must be "flattened". The values are compressed into a long vector or a column of sequentially ordered numbers.
Fully Connected Layer
The final layers of the CNN are densely connected layers, or an artificial neural network (ANN). The primary function of the ANN is to analyze the input features and combine them into different attributes that will assist in classification. These layers are essentially forming collections of neurons that represent different parts of the object in question, and a collection of neurons may represent the floppy ears of a dog or the redness of an apple. When enough of these neurons are activated in response to an input image, the image will be classified as an object.
 

The error, or the difference between the computed values and the expected value in the training set, is calculated by the ANN. The network then undergoes backpropagation, where the influence of a given neuron on a neuron in the next layer is calculated and its influence adjusted. This is done to optimize the performance of the model. This process is then repeated over and over. This is how the network trains on data and learns associations between input features and output classes.
The neurons in the middle fully connected layers will output binary values relating to the possible classes. If you have four different classes (let's say a dog, a car, a house, and a person), the neuron will have a "1" value for the class it believes the image represents and a "0" value for the other classes.
The final fully connected layer will receive the output of the layer before it and deliver a probability for each of the classes, summing to one. If there is a 0.75 value in the "dog" category, it represents a 75% certainty that the image is a dog.
The image classifier has now been trained, and images can be passed into the CNN, which will now output a guess about the content of that image.
Creating the Model
Creating the neural network model involves making choices about various parameters and hyperparameters. You must make decisions about the number of layers to use in your model, what the input and output sizes of the layers will be, what kind of activation functions you will use, whether or not you will use dropout, etc.
Learning which parameters and hyperparameters to use will come with time (and a lot of studying), but right out of the gate there are some heuristics you can use to get you running and we'll cover some of these during the implementation example.
Training the Model
After you have created your model, you simply create an instance of the model and fit it with your training data. The biggest consideration when training a model is the amount of time the model takes to train. You can specify the length of training for a network by specifying the number of epochs to train over. The longer you train a model, the greater its performance will improve, but too many training epochs and you risk overfitting.
Choosing the number of epochs to train for is something you will get a feel for, and it is customary to save the weights of a network in between training sessions so that you need not start over once you have made some progress training the network.
Model Evaluation
There are multiple steps to evaluating the model. The first step in evaluating the model is comparing the model's performance against a validation dataset, a data set that the model hasn't been trained on. You will compare the model's performance against this validation set and analyze its performance through different metrics.
There are various metrics for determining the performance of a neural network model, but the most common metric is "accuracy", the amount of correctly classified images divided by the total number of images in your data set.

This code pattern covers the following aspects:
•	Dataset preparation for training and testing
•	Running notebook for image classification
Install the following dependencies via pip:
i. Tensorflow
pip install tensorflow
ii. Numpy
pip install numpy
iii. SciPy
pip install scipy
iv. OpenCV
pip install opencv-python
v. Pillow
pip install pillow
vi. Matplotlib
pip install matplotlib
vii. H5py
pip install h5py
viii. Keras
pip install keras
ix. ImageAI
pip install imageai 
Tensorflow
TensorFlow is a Python library for fast numerical computing created and released by Google. It is a foundation library that can be used to create Deep Learning models directly or by using wrapper libraries that simplify the process built on top of TensorFlow.
Numpy
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
Scipy
SciPy is a library that uses NumPy for more mathematical functions. SciPy uses NumPy arrays as the basic data structure, and comes with modules for various commonly used tasks in scientific programming, including linear algebra, integration (calculus), ordinary differential equation solving, and signal processing.

OpenCV
OpenCV (Open Source Computer Vision) is a library of programming functions mainly aimed at real-time computer vision. In simple language it is library used for Image Processing. It is mainly used to do all the operation related to Images.

Pillow
Pillow is a fork of PIL (Python Image Library), started and maintained by Alex  
Clark and Contributors. It was based on the PIL code, and then evolved to a better, modern and more friendly version of PIL. It adds support for opening, manipulating, and saving many different image file formats Pillow offers several standard procedures for image manipulation. These include:
per-pixel manipulations, masking and transparency handling, image filtering, such as blurring, contouring, smoothing, or edge finding, image enhancing, such as sharpening, adjusting brightness, contrast or color, adding text to images and much more.
Matplotlib
Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
H5py
The h5py package is a Pythonic interface to the HDF5 binary data format.
Keras
Keras library used  library for deep learning in Python, especially for beginners. Its minimalistic, modular approach makes it a breeze to get deep neural networks up and running.
Image AI
ImageAI provides API to detect, locate and identify 80 most common objects in everyday life in a picture
