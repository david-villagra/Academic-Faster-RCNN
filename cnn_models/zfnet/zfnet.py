from __future__ import print_function
import numpy as np
import sys
import os
import argparse
current_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(current_dir, "../../utils/"))
import load_data_cifar100 as l
import torch
import torchvision
import torchvision.transforms as transforms


d = l.Cifar_data()
d.extract_cifar_data()

