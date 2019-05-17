import os
import numpy as np
import torch
# from torch import Dataset, Dataloader
from config import cfg
# from scipy import misc
import glob
import matplotlib.pyplot as plt


def preprocess_image(image):
    if cfg.USE_ZERO_MEAN_INPUT_NORMALIZATION is True:
        image = (image / 255.0 * 2.0 - 1.0)
    else:
        image /= 255.0
    return image


def read_annotation_file(filename,tp):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]
    # anno = np.array([])
    if tp == 'type' or tp == 'all':
        anno['type'] = np.array([cfg.RELABEL(x[0].lower()) for x in content])
    elif tp == 'position' or tp == 'all':
        anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
        anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
        anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
        anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    return anno


def lbl_to_num(corpus, labels):

    digit = np.array([])
    for l in labels:
        digit.append(corpus(l))
    return digit


def loadkitti(im_path, lbl_path=None):
    # output contains imdb, lbls

    images = np.array([])
    for im in im_path:
        if im.endswith('.png'):
            images.append(preprocess_image(plt.imread(im)))

    if not lbl_path is None:
        labels = np.array([])
        for l in lbl_path:
            if l.endswith('.txt'):
                labels.append(read_annotation_file(l, 'all'))

    # imdb = np.array([([]), ([]), ([]), ([]), ([])], dtype=[('image', 'float'), ('posX', 'float'),
    #                                                        ('posY', 'float'), ('label', labels['type'].dtype.name),
    #                                                        ('numlabel', 'int')])
    imdb = np.array([([]), ([]), ([]), ([])], dtype=[('image', 'float'), ('posX', 'float'),
                                                           ('posY', 'float'), ('numlabel', 'int')])
    imdb['image'] = images
     imdb['label'] = labels['type']
    imdb['posX'] = labels['2d_bbox_right']-labels['2d_bbox_left']
    imdb['posY'] = labels['2d_bbox_bottom'] - labels['2d_bbox_top']
    imdb['numlabel'] = lbl_to_num(cfg.CIFARLABELS_TO_NUM, labels['type'])

    return imdb

#########################################
# Problem is the labeling, not the values. Find out why array table can not be calles using 'posX' or 'label'