#test.py
#!/usr/bin/env python3

import argparse
#from dataset import *

#from skimage import io
from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_test_dataloader
from conf import settings


def test(net='zfnet', weights=settings.WEIGHT_PATH, gpu=True, b=16, s=True, output=settings.DATA_PATH):  #####################
    net = get_network(net=net, weights=weights, use_gpu=gpu, w=2, b=b, s=s, output=output)     #####################

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=2,
        batch_size=b,
        shuffle=s
    )

    net.load_state_dict(torch.load(weights), gpu)  # unpickle and make dict
    print(net)
    net.eval()

    # measurements
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    for n_iter, (image, label) in enumerate(cifar100_test_loader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = net(image)
        _, pred = output.topk(21, 1, largest=True, sorted=True)

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        correct_all += correct.sum()
        correct_5 += correct[:, :5].sum()
        correct_1 += correct[:, :1].sum()
        top_1 = 1 - correct_1 / len(cifar100_test_loader.dataset)
        top_5 = 1 - correct_5 / len(cifar100_test_loader.dataset)
        error = 1- correct_all / len(cifar100_test_loader.dataset)
        param_number = sum(p.numel() for p in net.parameters())
        # print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
        # print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
        # print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    return top_1, top_5, param_number

