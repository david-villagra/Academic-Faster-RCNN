# Training network branch
In this branch, the basic training of the specified network is done using the CIFAR-100 dataset.


## Prerequisites
Install pytorch with a (recommended) virtual environment that runs python3
```
pip3 install torch torchvision
```
After installing, you can check if both torch and CUDA are installed. Once that's done, the CIFAR100 dataloader that Pytorch includes has to be edited so that the datsets download the coarse labels. 

## Run the training
To train a network run the following command in the Linux command window:
```
python3 train.py -net <NETWORK>
```
If you want to use GoogLeNet, it is necessary to change the flag USE_ZFNET to 0 (
<code>
  $ USE_ZFNET = 0
</code>
) in the Settings file (in Conf folder). 

## Using GPU

As this branch was used for testing the CNN networks, GPU is disabled by default. For GPU usage to be enabled, change the flag US_GPU to 1 (
<code>
  $ USE_GPU = 1
</code>
) in Settings file. 
