import os
import numpy as np
import torch
# from torch import Dataset, Dataloader
from conf import settings
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
    # if settings.USE_ZERO_MEAN_INPUT_NORMALIZATION is True:
        # image = (image / 255.0 * 2.0 - 1.0)
    #else:
        # image /= 255.0
    return image


def read_annotation_file(filename,tp):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip().split(' ') for x in content]
    # anno = np.array([])
    anno = {}
    if tp == 'type':
        anno['type'] = np.array([settings.RELABEL[x[0].lower()] for x in content])
    elif tp == 'position':
        anno['2d_bbox_left'] = np.array([float(x[4]) for x in content])
        anno['2d_bbox_top'] = np.array([float(x[5]) for x in content])
        anno['2d_bbox_right'] = np.array([float(x[6]) for x in content])
        anno['2d_bbox_bottom'] = np.array([float(x[7]) for x in content])
    elif tp == 'all':
        anno['type'] = np.array([settings.RELABEL[x[0].lower()] for x in content])
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



    print(im_path)
    if im_path.endswith('.png'):
        print(plt.imread(im_path))
        images = preprocess_image(plt.imread(im_path))

    if lbl_path is not None:
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
    # np.append(imdb.numLabel, lbl_to_num(settings.CIFARLABELS_TO_NUM, labels['type']))
    imdb.label = labels['type']
    imdb.X = labels['2d_bbox_right']-labels['2d_bbox_left']
    imdb.Y = labels['2d_bbox_bottom'] - labels['2d_bbox_top']
    imdb.left = labels['2d_bbox_left']
    imdb.top = labels['2d_bbox_top']
    imdb.right = labels['2d_bbox_right']
    imdb.bottom = labels['2d_bbox_bottom']
    imdb.numLabel = lbl_to_num(settings.CIFARLABELS_TO_NUM, labels['type'])
    #print(images.size)
    #plt.interactive(False)
    #plt.imshow(images)
    #plt.title(str(imdb.label))
    #plt.show()
    return imdb


# not used and adapted so for
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep