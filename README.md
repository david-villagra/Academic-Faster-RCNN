# Academic implementation for Faster R-CNN
The code in this repository is a simplified version of the Faster R-CNN algorithm for object detection, being implemneted with the GoogleNet and ZFNet CNNs for classification.

For further theoretical details and the achievements obtained by this method, please refer to the final report done for this project, available [here](https://github.com/david-villagra/Academic-Faster-RCNN/blob/master/academic_faster_rcnn.pdf)

## Prerequisites
Install pytorch with a (recommended) virtual environment that runs python3
```
pip3 install torch torchvision
```
After installing, you can check if both torch and CUDA are installed. Once that's done, the CIFAR100 dataloader that Pytorch includes has to be edited so that the datsets download the coarse labels. 

## Basic structure

This method uses the basis of the [Faster-RCNN](https://arxiv.org/abs/1506.01497?context=cs.CV) architecture, combined with a Deep Convolutional Neural Network + Fully Connected layers, in order to achieve multiple object detection with real-time performance, on outdoor road-like environments.

The architecture takes as input an image that may have multiple objects in it. The image is then cropped in squared sections and the pieces are max pooled and sent as input to a CNN, that has previously been trained with images from the CIFAR-100 dataset (using coarse labels). The CNN Architecture is loosely coupled into the architecture, so it can be changed. For this project, two CNNs were implemnted: [GoogLeNet](https://research.google/pubs/pub43022/) and [ZFNet](https://arxiv.org/abs/1311.2901).

## Run the training phase
The code is prepared to use different training parameters for the network. Therefore, the first thing to do is edit the files test_cases_zfnet.py or test_cases_googlenet.py in order to have the desired parameters. Once the parameters are set, rfor the ZFNet CNN run
```
python3 test_cases_zfnet.py
```
And for the GoogLeNet CNN run
```
python3 test_cases_googlenet.py
```

## Test the anchor generation
The anchor generation is tested usig the original images and labels from the Kitti dataset. This way, after cropping the image in several parts, the pieces that are close enough to the ground truth label are kept. This code outputs the cropped images (32 x 32 x 3) that fulfill the requirements with respect to the ground truth labels

```
python3 evaluate_glob.py
```
