# Furniture Detection
Programs and Deep Learning Models for detecting Indoor Furniture

Furniture detection involves the object detection technology to predict various classes of indoor furnitures. There are 10 classes detected in this experiment, starting from bed, bookcase, chair, chest of drawers, cabinet, infant bed, stool, wardrobe, sofa bed, and table. The detection process utilize deep learning method using YOLOv8 architecture. There are 5 different pre-trained models to be used in this experiment, the n, s, m, l, and x version of YOLOv8

# Training Method
The training process begins with installing the necessary libraries and downloading the prepared dataset using roboflow. Since the dataset has been through some pre-processing and augmentation in roboflow, hence in the training phase, the datasets are directly trained using YOLOv8. The training process will be carried out by tuning the best values of several hyperparameters in terms of accuracy and computation. These hyperparameters include batch size, optimizer, image size, activation function, mosaic augmentation, and the number of epochs, as well as the depth and width of the architecture that will be used. The goal achieved is when the model has a high mAP value and does not occur overfitting or underfitting and achieves real-time computing speed. 

![image](https://github.com/arfashaha/FurnitureDetection/assets/64786796/d6bc228a-acfa-4140-98d0-f7754232c7ea)

Once the training has been finished, the next process is converting the model from pytorch format into other format for benchmarking. This is necessary to know which format represents the most optimal model in terms of accuracy and inference time. There are 4 formats which we used here, Pytorch, Torchscript, ONNX, and TensorRT. The benchmarking is done by using the validation data on the training dataset

# Dataset Analysis

There are a total of 10,372 images with 17,096 annotations, comprising 10 classes. The range of image resolutions captured for training varies from 0.41 Mega Pixels to 8.29 Mega Pixels. The comparison of data acquisition results is depicted in Figure 3.1. This graph represents the annotation count, revealing that the classes "Chair" and "Table" exhibit the highest frequency of occurrences, followed by "Sofa bed," "Stool," "Bookcase," and "Cabinet." The four classes with the lowest annotation counts are "Chest of Drawers," "Bed," "Infant Bed," and "Wardrobe," primarily due to a lower average annotation count per image

![image](https://github.com/arfashaha/FurnitureDetection/assets/64786796/44f72423-f8b3-4468-9a1a-035e8643c5db)

# mAP results for each models

These results are based on the validation data, means there are several similar images.
| Model | Precision | Recall  | mAP | Inference Time  |
| ------------- |:-------------:| -----:|:-----:| -----:|
| M | 95.1% | 88.1% | 93.8% | 2.0 ms |
| L | 95.7% | 88.0% | 94.0% | 3.4 ms |
| X | 96.6% | 88.5% | 94.4% | 4.8 ms |

# Testing Results on Completely Unseen Data

## Confusion Matrix

![image](https://github.com/arfashaha/FurnitureDetection/assets/64786796/4494cf8a-6dd0-4f60-b57c-bc524bb2fdd0)

## Overall mAP and mAP for each category

![image](https://github.com/arfashaha/FurnitureDetection/assets/64786796/b86ef5a6-0b7a-4fad-bd93-801ab4ebda08)

# How to Use

Users can directly download the models that they want to test, then run the Simple_Inference.py to evaluate the models.

There are various models that are provided here in Pytorch format (.pt). The ONNX and TensorRT models are not provided since they are hardware specifics, hence user has to convert the desired models from Pytorch to ONNX or TensorRT using the Converter.py provided above.

# Results Example

## Results in NVIDIA GeForce RTX 3050Ti

![image](https://github.com/arfashaha/FurnitureDetection/assets/64786796/246371f0-7514-440c-bd87-2dcb53d72b13)

## Results in NVIDIA Jetson Xavier

![Model M Inference Xavier](https://github.com/arfashaha/FurnitureDetection/assets/64786796/5c9432a1-812b-4ef5-a6e0-0ac12e83e6d2)
