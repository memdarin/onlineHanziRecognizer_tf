## OnlineHanziRecognizer

#### Start tensorBoard:

>tensorboard --logdir=./log/ --port=8090 --host=127.0.0.1
In a browser , go to : http://localhost:8090/

#### Generate the training and test dataset:

> python ImageDatasetGeneration.py

#### Start training:

In Pycharm Python code editor, 
 Script path: PATH_TO_PROJECT\OnlineHanziRecognizer\cnn.py
 Parameters: --mode=training
 Working directory : PATH_TO_PROJECT\OnlineHanziRecognizer

#### Launching the Web server :

In Pycharm Python code editor, 
 Script path : PATH_TO_PROJECT\OnlineHanziRecognizer\webapp\app.py
 Working directory : PATH_TO_PROJECT\OnlineHanziRecognizer\webapp
then start the project.
In a browser , go to : http://localhost:5000/


