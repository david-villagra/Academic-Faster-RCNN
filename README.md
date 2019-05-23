# Academic implementation for Faster R-CNN
The code in this repository is a simplified version of the Faster R-CNN algorithm for object detection.

## Prerequisites
Install pytorch with a (recommended) virtual environment that runs python3
```
pip3 install torch torchvision
```
After installing, you can check if both torch and CUDA are installed. Once that's done, the CIFAR100 dataloader that Pytorch includes has to be edited so that the datsets download the coarse labels. 

## Basic structure

The algorithm takes as input an image that may have multiple objects in it. The image is then cropped in squared sections and the pieces aremax pooled and sent as input to a CNN that has previously been trained with images from the CIFAR-100 dataset (using coarse labels)

## Run the training phase
The code is prepared to use different training parameters for the network. Therefore, the first thing to do is edit the files test_cases_zfnet.py or test_cases_googlenet.py in order to have the desired parameters. Once the parameters are set, rfor the ZFNet CNN run
```
python3 test_cases_zfnet.py
```
And for the GoogLeNet CNN run
```
python3 test_cases_googlenet.py
```

