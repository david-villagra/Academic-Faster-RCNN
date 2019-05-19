import torch
# from torch.utils.data import Dataset, Dataloader
import numpy as np
from conf import settings
from collections import namedtuple
from utils.kitti_tools import Image
import matplotlib.pyplot as plt


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy


def RoI_pooling(feature_map, H = 32, W = 32):
    # Inputs:
    # feature_map: matrix nxnx3 (cropped image)
    # H and W are the desired dimensions of the output default setting:32x32

    # Output: A WxHx3 matrix
    for i in range(feature_map.shape[2]): # Should be 3 (RGB)
        x1 = 0  # RoI_matrix[i, 1]
        x2 = feature_map.shape[0]  # RoI_matrix[i, 3]
        y1 = 0  # RoI_matrix[i, 2]
        y2 = feature_map.shape[1]  # RoI_matrix[i, 4]
        RoI_map[:, :, i] = feature_map[x1:x2, y1:y2]
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        height = h // H
        width = w // W
        for j in range(w) : # Should be n
            end_w = width * (j + 1) - 1
            if j == (W - 1):
                end_w = w - 1
            for k in range(H) : # Should be n
                end_h = height * (k + 1) - 1
                if k == (H - 1) :
                    end_h = h - 1
                RoI_reduced[j, k, i] = np.amax(RoI_map[j * width:end_w, k * height:end_h, i])

    return RoI_reduced


def checkOverlap(anchor, origIm, dist):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    w1 = settings.GTOVERLAP_CORE_THRES
    w2 = settings.GTOVERLAP_AREA_THRES
    dist = dist*w1
    #xOrig = origIm['posX']
    #yOrig = origIm['posY']
    labOrig = origIm.label
    top = origIm.top
    left = origIm.left
    bottom = origIm.bottom
    right = origIm.right
    xOrig = origIm.X
    yOrig = origIm.Y  # yUp + (yDown-yUp)/2

    x1 = anchor.left
    x2 = anchor.right
    y1 = anchor.top
    y2 = anchor.bottom

    x = x1 + (x1-x2)/2
    y = y1 + (y1-y2)/2
    r = Rectangle(x1, y1, x2, y2)

    for i in range(len(xOrig)):
        rOrig = Rectangle(left[i], top[i], right[i], bottom[i])
        if abs(xOrig[i]-x) <= dist and abs(yOrig[i]-y) <= dist:
            if (area(r, r)/area(rOrig, rOrig) >= 1-w2) and (area(r, r)/area(rOrig, rOrig) <= 1+w2):
                label = settings.RELABEL(labOrig[i])
        else:
            label = "background"

    return label


def getAnchors(imdb, ratios, is_fix, stride):
    # anchdb contains anchors, anch_ctrs, anch_lbls
    if is_fix is True:
        scale = imdb.image.size

    anchdb = Image()
    for i in range(len(imdb.image)):
        for r in ratios:
            cntr = r/2.0  # just for rectangular anchors
            for ix in range(floor((scale[0]-r)/stride)):
                for iy in range(floor((scale[1]-r)/stride)):
                    cX = cntr+ix*r
                    cY = cntr+iy*r
                    np.append(anchdb.X, cX)
                    np.append(anchdb.Y, cY)
                    np.append(anchdb.left, cX-cntr)
                    np.append(anchdb.top, cY-cntr)
                    np.append(anchdb.right, cX+cntr)
                    np.append(anchdb.bottom, cY+cntr)
                    img_temp = img
                    img_temp = img_temp[cntr-(ix+1)*r:cntr+(ix+1)*r][cntr-(iy+1)*r:cntr+(iy+1)*r][0:2]
                    # roi_matrix = np.transpose(np.array([0, r, 0, r]))
                    np.vstack(anchdb.image, RoI_pooling(img_temp, 32, 32))
                    label = checkOverlap(anchdb, img, dist=cntr)  #################### to change
                    np.append(anchdb.label, label)
                    np.append(anchdb.numlabel, settings.CIFARLABELS_TO_NUM(label))
                    plt.interactive(False)
                    plt.imshow(img_temp)
                    plt.title(label)
                    plt.show()
    return anchdb


def randomBackground():
    i = np.random.randint(2, size=1)
    weight = np.random.uniform(0, 1, size=(1, 1, 3))
    weight = np.tile(weight, (32, 32, 1))
    if i == 0:
        image = np.random.uniform(0, 1, (32, 32, 3))
        image += weight
    elif i == 1:
        image = np.random.beta(np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 1), (32, 32, 3))
        image += weight
    return image
