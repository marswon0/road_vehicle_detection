# Road Vehicle Detection Through YOLOv5

## Introduction

In urban city, there is a gap between the ideal and actual commute time. With the help of autonomous vehicle, travelers can make a much better use of their commute 
time. This project focuses on the road vehicle detection problem which is part of the autonomous vehicle design. The performance of different YOLOv5 models were validated on a custom dataset that contains images of 5 types of vehicles.

The performance of the YOLOv5 model is directly proportional to the quality of the training data: the number of instances of each target object, the quality and consistency of the labeling. This project demonstrates the Average Precision (AP) of a single object class could be improved by using the proposed two-stage training technique; in the 2nd training stage, a well labeled dataset with less than 30 images improved the detection accuracy for trucks by 8%.

## Reference

For more details please check out [the report paper written for this project](https://github.com/marswon0/road_vehicle_detection/blob/main/assets/paper/Road%20Vehicle%20Detection%20Through%20YOLOv5.pdf).


## Usage

### Address of the dataset

Change the file addresses used in '/data/custom_dataset.yaml' and '/data/custom_truck.yaml' to the actual location of the datasets.

### Standard training

Consider to use the following commands to train the model:
    
    python train.py --img 640 --batch 16 --epochs 50 --data custom_dataset.yaml --weights yolov5s.pt

### Two-step training

1.Train the selected model on the first dataset using the command below :

    python train.py --img 640 --batch 16 --epochs 50 --data custom_truck.yaml --weights yolov5s.pt

2.In the "project\runs\train\exp\weights" folder, copy the "best.pt" file
3.Paste the "best.pt" file under the project folder
4.Train the selected model with the next dataset

    python train.py --img 640 --batch 16 --epochs 50 --data custom_vehicle.yaml --weights best.pt

### YOLOv5 Model

For more details about the YOLOv5 model, please visit the [official YOLOv5 Github page](https://github.com/ultralytics/yolov5).

## Model Architecture

### Model Backbone

The YOLOv5 architecture is showned on the figure below. The module backbone is responsible for extracting features and patterns from input images. The model backbone uses the CSP-Darknet53 network structure that consists of 5 convolution layers, 3 C3 layers, and a Spatial Pyramid Pooling – Fast (SPPF) layer. Every convolution layer uses 2D convolution, 2D batch normalization, and SiLU activation function.

The C3 layer includes 3 convolution layers. The input received by the C3 layer will go through 2 convolution layers in parallel. The outputs of the two layers are concatenated, and then fed to the 3rd convolution layer as input.

At each convolution layer in the model backbone, the stride size is 2 (not including the C3 layer). After the input image propagate through the entire backbone model, the channel size will increase from 3 channels (usually input image contains RGB channels) to 1024 channels. Meanwhile, the image size will be downsampled to 1/32 of the original size. 

At the end of the model backbone, a SPPF layer is added to make sure the model is more robust towards object deformations. The output of the SPPF layer will be fed to the first convolution layer in the model neck as input. 

### Model Neck

The model neck generates the feature pyramid that helps to generalize scaling for objects with the same label, as well as identifying objects not seen in the training set. YOLOv5 model uses enhanced bottom-up paths to shorten the travel path between features in lower layers and the deeper layers. Therefore, the localization information can be accessed by the topmost layers more accurately.

### Model Head

The model head performs the final prediction on objects that includes bounding box locations, object labels, and probability for each object classes.

<img src="/assets/images/model.JPG" width="650" height="850">

- i: input channel
- o: output channel
- k: is kernel size
- s: stride size
- n: number of convolution blocks

## Dataset

The [custom dataset](https://b2n.ir/vehicleDataset) used in this project has 1321 images that include the following classes: car, motorcycle, truck, bus, bicycle. Multiple objects can appear in the same image.


## Fine-tuning

The YOLOv5 model is pretrained with the COCO dataset that has more than 200K labelled images, 1.5 million object instances, and 80 object categories. Although the COCO dataset includes images for all type of vehicles appeared in the custom dataset, the model had a poor performance in recognizing motorcycle, truck, bus, and bicycle at a 0.157 mAP without fine tuning. Most of the misclassified objects were recognized as either car or background.

<img src="/assets/images/fine_tuning.JPG">

## Performance of different YOLOv5 models

YOLOv5 models with different sizes are trained and validated using the same dataset. The training epoch on each model is set to 50. The performance of each model is concluded in table below.

<img src="/assets/images/diff_models.JPG" width="600" height="160">


## Two-stage Training Strategy

There are multiple factors that affect the detection accuracy of the YOLOv5 models. Samples contained in the dataset is mostly cars rather other types of vehicles. This indicates imbalance data between object classes. The number of cars vs. other vehicles is at a 10~20:1 ratio. This imbalance causes the model to focus on the car detection rather than other vehicles.

<img src="/assets/images/imba.jpg" width="600" height="600">

To increase the accuracy for recognizing a particular type of vehicle, a 2-stage training method was adapted. In stage one, the YOLOv5s model will be trained with only truck samples (15 images) as an independent dataset for 50 epochs. In state two, the YOLOv5s model will be further trained on the same dataset (1321 images) foranother 15 epochs. The results showed the AP for truck object increased from 0.438 to 0.5 after the 2-stage training. The overall model performance increased slightly. 

<img src="/assets/images/2_stage.JPG">

## Vehicle Detection Results

<img src="/assets/images/result2.jpg">

<img src="/assets/images/result.jpg">
