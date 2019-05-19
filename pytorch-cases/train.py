# train.py
#!/usr/bin/env	python3


import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
#from dataset import *
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR


def train(epoch, args, net, cifar100_training_loader, warmup_scheduler, loss_function, optimizer):

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= args.warm:
            warmup_scheduler.step()

        images = Variable(images)
        labels = Variable(labels)

        if args.use_gpu is True:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        #for name, para in last_layer.named_parameters():
            #if 'weight' in name:
            #    writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            #if 'bias' in name:
            #    writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

        #update training loss for each iteration
        #writer.add_scalar('Train/loss', loss.item(), n_iter)

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        #writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


def eval_training(epoch, net,cifar100_test_loader, args, loss_function):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:
        images = Variable(images)
        labels = Variable(labels)
        if args.use_gpu is True:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    #add informations to tensorboard
    #writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    #writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    acc = correct.float() / len(cifar100_test_loader.dataset)

    return acc, loss, test_loss


def train_net(args):
    net = get_network(args)
        
    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        args,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.dat_sh
    )
    
    cifar100_test_loader = get_test_dataloader(
        args,
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=args.w,
        batch_size=args.b,
        shuffle=args.dat_sh
    )

    if args.loss is 'cel':
        loss_function = nn.CrossEntropyLoss()
    elif args.loss is 'smoothl1':
        loss_function = nn.SmoothL1Loss()
    elif args.loss is 'mlml':
        loss_function = nn.MultiLabelMarginLoss()

    if args.optim is 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=args.wdecay)
    elif args.optim is 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr_init, weight_decay=args.wdecay)

    if args.lr_fct is 'MSscheduler':
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.mil, gamma=0.2) #learning rate decay
    elif args.lr_fct is 'cyclic':
        optimizer = optim.SGD(net.parameters(), lr=args.lr_init, weight_decay=args.wdecay)
        train_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.max_lr,
                                                      cycle_momentum=True)

    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH+'/'+args.testname, args.net, settings.TIME_NOW)

    #use tensorboard
    args.output = args.output + '/' + args.testname
    if not os.path.exists(args.output+'/log'):
        os.makedirs(args.output + '/log')
    #writer = SummaryWriter(log_dir=os.path.join(
    #        settings.LOG_DIR, args.net, settings.TIME_NOW))

    if args.use_gpu is True:
        input_tensor = torch.Tensor(12, 3, 32, 32).cuda()
    else:
        input_tensor = torch.Tensor(12, 3, 32, 32)
    # writer.add_graph(net, Variable(input_tensor, requires_grad=True))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    acc = []
    loss = []
    test_loss = []

    for epoch in range(1, args.num_iter):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        train(epoch, args, net, cifar100_training_loader, warmup_scheduler, loss_function, optimizer)
        with torch.no_grad():
            acc_temp, loss_temp, test_loss_temp = eval_training(epoch, net, cifar100_test_loader, args, loss_function)
        acc.append(acc_temp)
        loss.append(loss_temp)
        test_loss.append(test_loss_temp)
        torch.cuda.empty_cache()
        #start to save best performance model after learning rate decay to 0.01 
        #if epoch > settings.MILESTONES[1] and best_acc < acc:

        if settings.SAVE_WEIGHTS is True:
            if epoch > args.mil[1] and best_acc < acc:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    #writer.close()

    results = [acc, loss, test_loss]
    return results
