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
    anno = {}
    if tp == 'type' or tp == 'all':
        anno['type'] = np.array([x[0].lower() for x in content])
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
    imdb = np.array([(), (), (), ()], dtype=[('image', 'float'), ('label', 'S'), ('vallabel', 'ind')])
    im_files = os.listdir(im_path)

    images = []
    for im in im_files:
        if im.endswith('.png'):
            images.append(preprocess_image(plt.imread(im)))
    #images = [preprocess_image(plt.imread(f)) in sorted(f) in im_files if f.endswith('.png')]

    if not lbl_path is None:
        lbl_files = os.listdir(lbl_path)
        labels = []
        for l in lbl_files:
            if l.endswith('.txt'):
                labels.append(read_annotation_file(l, 'type'))
        # labels = [read_annotation_file(f, 'type') in f in lbl_files if f.endswith('.txt')]  # put 'type' or 'position' or 'all'
    for i in range(len(labels)-1):
        if not labels[i] in cfg.LABELS_KEEP:
            labels[i] = 'misc'

    corpus = dict((name, index) for index,name in enumerate(np.unique(np.array(labels))))

    imdb['image'] = np.array(images)
    imdb['label'] = np.array(labels)
    imdb['vallabel'] = np.array(lbl_to_num(corpus, labels))

    return imdb

