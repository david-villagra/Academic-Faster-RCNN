#test.py
#!/usr/bin/env python3

import argparse
import numpy as np
#from dataset import *

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from conf import settings
from utils import get_network, get_test_dataloader
from conf import settings

from scipy.ndimage import imread as jpg_to_numpy 
from scipy.special import softmax
import matplotlib.pyplot as plt
from PIL import Image

def test_one(args):
    net = get_network(args)
    if args.use_gpu is False:
        net.load_state_dict(torch.load(args.path_pth, map_location='cpu'))
    else:
        net.load_state_dict(torch.load(args.path_pth), gpu = args.use_gpu)
    #print(net)
    net.eval()
    image = jpg_to_numpy(args.path_img)
    img_PIL = Image.fromarray(image)
    #print(np.shape(image))
    #image = Image.open(args.path_img)
    if (args.net is 'zfnet'):
        transform_test = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomResizedCrop(224-4-4),
            transforms.Pad(4),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])
    elif (args.net is not 'zfnet'):
        transform_test = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(settings.CIFAR100_TRAIN_MEAN, settings.CIFAR100_TRAIN_STD)
        ])

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #image = torch.from_numpy(image).float().to(device)
    img_tensor = transform_test(img_PIL).float()
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = Variable(img_tensor)
    #image = Variable(image, requires_grad=True)
    output = net(img_tensor)
    out_np = output.detach().numpy()
    probs_out = softmax(out_np)
    prob_best = np.max(probs_out)*100.0
    print(prob_best)
    best_out = settings.NUM_TO_CIFARLABELS[np.argmax(out_np)]
    #image.show()
    plt.imshow(image)
    plt.title('Identified as: '+ best_out + ', with probability: ' + str(np.around(prob_best,decimals=2)) + '%')
    plt.show()
    return 0

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


