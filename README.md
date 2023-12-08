<br> ️*_This utilizes Kazuhito Takehashi's hand gesture recognition: [original repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)._*
<br> 
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

This repository contains the following contents.
* Sample program
* Hand sign recognition model(TFLite)
* Finger gesture recognition model(TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

# Hand Gesture/Sign Recognition
This sample program recognizes hand signs and finger gestures with a simple MLP using the detected key points. In addition, it builds its model by utilizing a linear stack of layers, which takes the output of a previous layer and uses it as an input for the next in a sequential fashion.
Input layer: Accounts for 21 features, which also account for sets of 2 for each feature
Output layer:  Regularization technique, Dropout, to randomly set 20% of inputs to 0. This reduces dependency and promotes redundancy.

# Redundancy
Redundancy, in this context, means that multiple pathways in the network can contribute to the same or similar features. This can improve the generalization of the model because it becomes less sensitive to the precise configuration of neurons. The model is better equipped to recognize features in various forms or contexts.

# Classes
0. Open
1. Close
2. Point
3. Ok
4. Thumbs Up
5. Three Fingers
6. Four Fingers
   
# Confusion Matrix
Confusion Matrix           |  Classification Report
:-------------------------:|:-------------------------:
<img src="imgs/confusionMatrix2.png" height="300px" width="400px"><br><be> |  <img src="imgs/confusionMatrix.png" height="300px" width="400px"><br><be>

A **confusion matrix** is a table that is often used to evaluate the performance of a classification model on a set of data for which the true values are known. 

**Precision:** The ratio of correctly predicted positive observations to the total predicted positives (TP / (TP + FP)). It measures the accuracy of positive predictions. 

**Recall:** The ratio of correctly predicted positive observations to all observations in the actual class (TP / (TP + FN)). It measures the ability of the model to capture all the relevant cases. 

**F1 Score:** The harmonic mean of precision and recall, providing a balance between the two metrics. 

**Support:** The number of actual occurrences of each class in the specified dataset.

## Analysis
**Class-wise Performance:**
> Classes 0, 1, 2, and 4 have high precision, recall, and F1 scores, suggesting good performance.
> Class 5 has lower precision, recall, and F1-score, indicating potential challenges in predicting this class accurately.
> Class 6 has lower precision and F1-score, but a higher recall, suggesting the model is better at capturing instances of this class.
> Though Class 5 and 6 have less support, a plethora of trials were conducted where they presented additional support; however, the same issue occurred. 

**Overall Performance:**
> The model has an overall accuracy of 93%, which is a good sign.
> Macro-average and weighted-average metrics provide an overall assessment. They are close, indicating a balanced dataset.

**Recommendations:**
> Exploring techniques such as adjusting class weights, tuning hyperparameters, or using different algorithms to improve the model's performance, especially for challenging classes.
> Experiment with various fractions for regularization

# Errors 
Three Fingers              |  Four Fingers
:-------------------------:|:-------------------------:
<img src="imgs/Three Fingers.png" height="300px" width="400px"><br><be> |  <img src="imgs/Four Fingers.png" height="300px" width="400px"><br><be>

**Note:** Recognition of Four Fingers occurred reliably only when positioned in a manner resembling the top of a circle.

## Feature Design
The features used to train the model may not capture the nuances that differentiate classes 5 and 6, despite regularization.

## Model
The model architecture may not be sufficiently complex to learn the intricate patterns that differentiate between classes 5 and 6. 

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```
### app.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
This directory stores files related to finger gesture recognition.<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

The model is trained on a dataset represented by X_train and y_train, which contain input features and corresponding labels, respectively. The training process involves iterating over the entire dataset for a specified number of epochs (in this case, 1000 epochs) and updating the model's weights based on the optimization of a chosen loss function. The training is performed in batches of 128 samples at a time, enhancing computational efficiency and memory usage. 

### Hand sign recognition training
#### 1.Learning data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
In the initial state, three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2).<br>
If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" <br>and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### Finger gesture recognition training
#### 1.Learning data collection
Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/point_history_classifier/point_history.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>
In the initial state, 4 types of learning data are included: stationary (class ID: 0), clockwise (class ID: 1), counterclockwise (class ID: 2), and moving (class ID: 4). <br>
If necessary, add 5 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

#### 2.Model training
Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 4" and <br>modify the label of "model/point_history_classifier/point_history_classifier_label.csv" as appropriate. <br><br>

#### X.Model structure
The image of the model prepared in "[point_history_classification.ipynb](point_history_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
The model using "LSTM" is as follows. <br>Please change "use_lstm = False" to "True" when using (tf-nightly required (as of 2020/12/16))<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# Reference
* [MediaPipe](https://mediapipe.dev/)

# Translation and other improvements
Nikita Kiselov(https://github.com/kinivi)
 
# License 
Hand Gesture Recognition via Mediapipe [Apache v2 license](LICENSE).
