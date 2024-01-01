# Recognition of Emotion from Images Using Deep Learning

## Abstract
This project explores human emotion recognition from images using deep learning, focusing on CNN architectures including a baseline CNN, a modified VGGNet, and a pre-trained ResNet-50 with transfer learning. The FER-2013 dataset, containing grayscale facial expression images, was used for training and evaluation.

## Installation
Flask==3.0.0
gunicorn==20.1.0
tensorflow==2.8.0
numpy==1.22.3
Pillow==9.0.1
matplotlib==3.5.1
seaborn==0.11.2
Werkzeug==3.0.0

## Dataset
FER-2013: Contains 35887 grayscale images of faces across seven emotion classes (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)

## Models
1. Baseline Model: Custom CNN
2. VGGNet Modified CNN
3. ResNet-50 with Transfer Learning

## Results
The models were evaluated based on accuracy, loss, and confusion matrices, with ResNet-50 showing superior performance (71% Test Accuracy)
