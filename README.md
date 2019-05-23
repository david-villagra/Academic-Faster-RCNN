+# Test with one image
The code in this repository is used to test the performance of the trained weights of the CNN when a cropped part of a Kitti image is sent as input.

## Prerequisites
Install pytorch with a (recommended) virtual environment that runs python3
```
pip3 install torch torchvision
```
After installing, you can check if both torch and CUDA are installed. Once that's done, the CIFAR100 dataloader that Pytorch includes has to be edited so that the datsets download the coarse labels. 

## Basic structure

The network takes the pretrained weights and an image of size 32 x 32 x 3. The output is a vector of probabilities. The program also displays the image along with its estimated label and the probability that it represents the found label

## Run the code
To run the code, you must have an image saved in the /images folder of the correct size, and a trained model of the network that you want to use. The program will create a copy of the image with the label identified, and with the same name + "\_result". The following command runs the code:
```
python3 test_one_img.py
```
This will run the configuration by default, which is with googLeNet network. In order to make the code runnable, you may need to change the paths of both the input image, and the trained model that you want to use (as of now, either googleNet or ZFNet).

If interested in changing those variables:
```
python3 test_one_img.py -net <Network_name> -path_pth <complete_path_to_trained_model> -path_img <complete_path_to_input_image>
```

