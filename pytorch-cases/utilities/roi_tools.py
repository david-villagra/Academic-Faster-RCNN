import torch
# from torch.utils.data import Dataset, Dataloader
import numpy as np
from conf import settings
from collections import namedtuple
from utilities.kitti_tools import Image
import matplotlib.pyplot as plt
import skimage as sk
from scipy import ndimage


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy

def RoI_pooling(feature_map, H = 32, W = 32):
    #check = sk.measure.block_reduce(feature_map, (H, W, 3), np.max)
    check = ndimage.zoom(feature_map, zoom = (W/feature_map.shape[0],H/feature_map.shape[1],1))
    #print(check.shape)
    return check
    # Inputs:
    # feature_map: matrix nxnx3 (cropped image)

    # H and W are the desired dimensions of the output default setting:32x32
    # RoI_map = np.zeros((feature_map.shape[0], feature_map.shape[1], 3))
    # RoI_reduced = np.zeros((32, 32, 3))
    # # Output: A WxHx3 matrix
    # for i in range(feature_map.shape[2]): # Should be 3 (RGB)
    #     x1 = 0  # RoI_matrix[i, 1]
    #     x2 = feature_map.shape[0]  # RoI_matrix[i, 3]
    #     y1 = 0  # RoI_matrix[i, 2]
    #     y2 = feature_map.shape[1]  # RoI_matrix[i, 4]
    #
    #     w = abs(x1 - x2)
    #     h = abs(y1 - y2)
    #     height = h // H
    #     width = w // W
    #     #print(w)
    #     #print(height)
    #     #print(width)
    #     for j in range(W) : # Should be n
    #
    #         end_w = width * (j)
    #         # print(end_w)
    #         if j == (W - 1):
    #             end_w = w - 1
    #         for k in range(H) : # Should be n
    #             end_h = height * (k)
    #             print(end_h)
    #             if k == (H - 1):
    #                 end_h = h - 1
    #                 #print(end_h)
    #             RoI_reduced[j, k, i] = np.amax(RoI_map[(j-1) * width:end_w, (k-1) * height:end_h, i])
    #
    # return RoI_reduced


def checkOverlap(anchor, origIm, dist):
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    w1 = settings.GTOVERLAP_CNTR_THRES
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
    #print(xOrig)
    #print(anchor.X[-1])
    x1 = anchor.left[-1]
    x2 = anchor.right[-1]
    y1 = anchor.top[-1]
    y2 = anchor.bottom[-1]

    x = anchor.X[-1] # center of the anchor
    y = anchor.Y[-1] # center of the anchor
    r = Rectangle(x1, y1, x2, y2)
    label = []
    weight = []
    for i in range(len(xOrig)):
        rOrig = Rectangle(left[i], top[i], right[i], bottom[i])
        #print(abs(xOrig[i]-x))
        #print(abs(yOrig[i] - y))
        #print(dist)
        if (abs(xOrig[i]-x) <= dist) and (abs(yOrig[i]-y) <= dist):
            print("Test1")
            if (area(r, r)/area(rOrig, rOrig) >= 1-w2) and (area(r, r)/area(rOrig, rOrig) <= 1+w2):
                weight.append(np.abs(np.abs(area(r, r)/area(rOrig, rOrig))-w2))
                label.append(labOrig[i])  #(settings.RELABEL[labOrig[i]])
                print("Test2")
        #else:
        #    label.append("background")
    # print(str(label))
    bestlabel = "background"
    bestw = 1
    for i in range(len(weight)):
        if (weight[i]) < bestw:
            bestw = weight[i]
            bestlabel = label[i]
    return bestlabel


def getAnchors(imdb, ratios, is_fix, stride, imgnumber):
    # anchdb contains anchors, anch_ctrs, anch_lbls
    if is_fix is True:
        shape = imdb.image.shape
        #print(str(shape))
    anchdb = Image()
    testanch = Image()
    imcnt = 0
    for r in ratios:
        cntr = r/2.0  # just for rectangular anchors
        wid = r/2.0
        for ix in range(np.int(np.floor((shape[0]-r)/stride))):
            for iy in range(np.int(np.floor((shape[1]-r)/stride))):
                cX = cntr+ix*stride
                cY = cntr+iy*stride
                img_temp = imdb.image
                #print(np.shape(img_temp))
                # print(str(ix) + ' ' + str(iy))
                # print(wid)
                # print(np.int(cX-cntr))
                # print(np.int(cX+cntr))
                # print(np.int(cY-cntr))
                # print(np.int(cY+cntr))
                testanch.X = np.append(testanch.X, cX)
                # print(anchdb.X)
                testanch.Y = np.append(testanch.Y, cY)
                testanch.left = np.append(testanch.left, cX-cntr)
                testanch.top = np.append(testanch.top, cY-cntr)
                testanch.right = np.append(testanch.right, cX+cntr)
                testanch.bottom = np.append(testanch.bottom, cY+cntr)
                img_temp = img_temp[np.int(cX-cntr):np.int(cX+cntr),np.int(cY-cntr):np.int(cY+cntr),:3]
                # roi_matrix = np.transpose(np.array([0, r, 0, r]))
                testanch.image = RoI_pooling(img_temp, 32, 32)
                label = checkOverlap(testanch, imdb, dist=cntr)  #################### to change
                if label is not "background":
                    if imcnt is 0:
                        # print("True")
                        anchdb.image = testanch.image

                    else:
                        anchdb.image = np.dstack((anchdb.image, testanch.image))
                    imcnt = 1 + imcnt

                    anchdb.label.append(label)
                    anchdb.numLabel = np.append(anchdb.numLabel, settings.CIFARLABELS_TO_NUM[label])
                    anchdb.X = np.append(anchdb.X, cX)
                    # print(anchdb.X)
                    anchdb.Y = np.append(anchdb.Y, cY)
                    anchdb.left = np.append(anchdb.left, cX-cntr)
                    anchdb.top = np.append(anchdb.top, cY-cntr)
                    anchdb.right = np.append(anchdb.right, cX+cntr)
                    anchdb.bottom = np.append(anchdb.bottom, cY+cntr)
    print((anchdb.image.shape))
    print(len(anchdb.label))
    for l in range(len(anchdb.label)):
        if anchdb.label[l] is not "misc":                #plt.interactive(False)
            try:
                picture = anchdb.image[:,:,l*3:l*3+3]
                print(picture.shape)
                plt.imshow(picture)
                plt.title(anchdb.label[l])
                plt.draw()
                plt.savefig("testanchor"+str(imgnumber)+str(l)+".jpg")
            except:
                pass
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
