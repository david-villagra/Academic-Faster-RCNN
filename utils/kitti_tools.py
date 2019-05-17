import os
import numpy as np
import torch
# from torch import Dataset, Dataloader
from config import cfg
# from scipy import misc
import glob
import matplotlib.pyplot as plt


class Image:
    image = np.array([])
    label = np.array([])
    X = np.array([])
    Y = np.array([])
    top = np.array([])
    left = np.array([])
    bottom = np.array([])
    right = np.array([])
    numLabel = np.array([])

#
# class anchor:
#     image = np.array([])
#     label = np.array([])
#     X = np.array([])
#     Y = np.array([])
#     numLabel = np.array([])


def preprocess_image(image):
    if cfg.USE_ZERO_MEAN_INPUT_NORMALIZATION is True:
        image = (image / 255.0 * 2.0 - 1.0)
    else:
        image /= 255.0
    return image


def read_annotation_file(filename,tp):
    with open(cfg.LABLE_PATH + '/' + filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]
    # anno = np.array([])
    anno = {}
    if tp == 'type':
        anno['type'] = np.array([cfg.RELABEL[x[0].lower()] for x in content])
    elif tp == 'position':
        anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
        anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
        anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
        anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    elif tp == 'all':
        anno['type'] = np.array([cfg.RELABEL[x[0].lower()] for x in content])
        anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
        anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
        anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
        anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    return anno


def lbl_to_num(corpus, labels):

    digit = np.array([])
    for l in labels:
        np.append(digit, corpus[l])
    return digit


def loadkitti(im_path, lbl_path=None):
    # output contains imdb, lbls

    images = np.array([])
    for im in im_path:
        if im.endswith('.png'):
            images.append(preprocess_image(plt.imread(im)))

    if not lbl_path is None:
        labels = {}
        if lbl_path.endswith('.txt'):
            labels = read_annotation_file(lbl_path, 'all')
    # right now we just load one image
    imdb = Image()
    imdb.image = images
    # np.vstack(imdb.image, images)
    # np.append(imdb.label, labels['type'])
    # np.append(imdb.X, labels['2d_bbox_right']-labels['2d_bbox_left'])
    # np.append(imdb.Y, labels['2d_bbox_bottom'] - labels['2d_bbox_top'])
    # np.append(imdb.left, labels['2d_bbox_left'])
    # np.append(imdb.top, labels['2d_bbox_top'])
    # np.append(imdb.right, labels['2d_bbox_right'])
    # np.append(imdb.bottom, labels['2d_bbox_bottom'])
    # np.append(imdb.numLabel, lbl_to_num(cfg.CIFARLABELS_TO_NUM, labels['type']))
    imdb.label = labels['type']
    imdb.X = labels['2d_bbox_right']-labels['2d_bbox_left']
    imdb.Y = labels['2d_bbox_bottom'] - labels['2d_bbox_top']
    imdb.left = labels['2d_bbox_left']
    imdb.top = labels['2d_bbox_top']
    imdb.right = labels['2d_bbox_right']
    imdb.bottom = labels['2d_bbox_bottom']
    imdb.numLabel = lbl_to_num(cfg.CIFARLABELS_TO_NUM, labels['type'])
    return imdb

#########################################
# Problem is the labeling, not the values. Find out why array table can not be calles using 'posX' or 'label'