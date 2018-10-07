## OnlineHanziRecognizer

### The project :

 This project is a "online Handwritten chinese character recognizer" based on deep neural network.
 While the user handwrites a chinese character using the mouse, the computer recognizes the handwritten character and display the results. 
 Every time the user adds a stroke, the computer restarts the recognition process and updates the results.
 The results are a list of most probable candidate character.
 The engine is based on a deep neural network using tensorflow
 
#### Screenshots Web UI
![](https://github.com/itanghiu/onlineHanziRecognizer/blob/master/doc/OnlineHanziRecogWebUI.PNG)


### General principle of the chinese character recognition process

Th way the system recognizes the handwritten characters fllows the steps below:
  - the user draws the first stroke of the character. 
  - as soon as he lifts up the mouse pointer,the uncompleted image of the character is sent to the server,
  - the image is fed to the neural network who tries to recognize the character. It fails because most of the time, this uncompleted character does not correspond to an existing character.
 this process is repeated until the user adds the last stroke. This time, the image of the character wll be completed and the system will recognize the corresponding character.

### Generation of Training and testing dataset.

The dataset comes from the CASIA offline database (http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html). 
more precisely, the dataset is the "CASIA standard sample data" that everyone can download.
There exists a more complete CASIA dataset (3,895,135 character samples covering 7,185 characters ), unfortunately, as it is stated on the CASIA website:
"To avoid data misuse, we now license full databases to experienced researchers only". Which means that the academic world does not intend to share all its data with the rest of the world. 

The dataset used is the HWDB1.1 dataset. To download it :
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

It includes 3,755 Chinese characters and 171 alphanumeric and symbols.
To every character corresponds handwritten images from approximately 300 writers.
The  dataset contains a training set and a test set. 
Test set contains 60 randomly sampled images for each character,  and training set contains 240 image samples. 

In this project, the original training dataset is split into two parts: 
 - a training set (200 images for every character)
 - a validation set(approximately 40 images for every character).
 
#### Dataset generation
 
 The CASIA dataset contains x gnt files. Each gnt image contains y handwritten characters written by one writer.
 all the character images contained in one gnt file must be extracted as a png file and put into its corresponding directory.
the name of the directories (label) is an integer between 0 and 3755. The dictionary ImageDatasetGeneration.char_label_dictionary stores the correspondance between the character and the label.

The script ImageDatasetGeneration.py generates the dataset. To start it :
 > python PATH_TO_PROJECT\OnlineHanziRecognizer\ImageDatasetGeneration.py
 
 When the script completes, 895 000 files in 3755 folders will be generated.

### Code organization 

#### app.py : 
this module starts the web server and contains the REST API called by the browser to communicate with the server.

#### cnn.py : 
this module contains all the code related to the convolutional neural network and its training

#### Data.py : 
This python class manages the input data pipeline used for feeding the network with input data during the training phase.

#### utils.py : 
This module contains utility classes used by cnn.py.

### Deep neural network architecture.

This network uses :
 - 4 convolutional layers with filter size 3x3, stride = 1 and zero padding size= 1
 - 4 max pooling layers of size 2x2 and stride=2
 - 2 fully connected layers

This architecture seems to be a good compromise as adding additional covolutional layers only brings marginal gains. 

#### Learning Rate Scheduling

The idea of "Learning Rate Scheduling" is to start with a high learning rate and then reduce it once it stops making fast progress.
 That way, it is possible to get good solution faster than with the optimal constant learning rate.
The "exponential scheduling" has been chosen for this project with the following values
initial_learning_rate =2e-4
decay_rate=0.97
decay_steps=2000
the learning rate decreases according to the formula: η(t) = η0 10–t/r
The learning rate will drop by a factor of 10 every r steps.
 
#### Error function optimizer 

 the Adam optimizer is used in this project. It is an adaptive learning rate algorithm.
 It combines the ideas of RMSProp and Momentum optimization. 

#### Regularization

Regularization consists in avoiding overfitting the training set. The 2 techniques used in this project are :
 - early stopping
 - dropout

Early stopping : it consists in interrupting the training as soon as the performance on the validation set drops.
The way it is implemented in the project is to to regularly evaluate the model on a validation set every x steps.
 If it is better than the  previous snapshots then we save it as the "champion" model. The training is interrupted when the model performance stops improving.

Dropout: dropout refers to ignoring (they are not considered during forward or backward pass of a training step) certain set of neurons chosen randomly, during the training phase.
The dropout rate chosen is 0.8

the regularization is only used in the fully connected layers. It is not used can be used in the convolutional layers for the following reasons:
Since the convolutional layers do not do any prediction, there is no overfitting issues with them. They only  extract features. So adding regularization would hamper the process of extracting the features.

####  Weight initialization:

The weight fo the fully connected layers are initialized with the Xavier initialization. because, by default, the tensorflow fully_connected() function uses it.
