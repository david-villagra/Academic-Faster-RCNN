import torch
import numpy as np
from torch.utils.data import Dataset
from conf import settings
from utilities.kitti_tools import loadkitti
from utilities.roi_tools import getAnchors
import os
# from net import predict


if __name__ == '__main__':
    im_files = os.listdir(settings.IMDB_PATH)
    lbl_files = os.listdir(settings.LABEL_PATH)
    im_files = list(np.unique(np.array(im_files)))
    lbl_files = list(np.unique(np.array(lbl_files)))

    for i in range(len(im_files)):
        im_files[i] = settings.IMDB_PATH + '/' + im_files[i]
        lbl_files[i] = settings.LABEL_PATH + '/' + lbl_files[i]

    imscale_is_fix = True
    ratios = [128, 192]

    stride = 16
    for i in range(10):
        imdb = loadkitti(im_files[i], lbl_files[i], i*100)  # imdb contains imdb, lbls, positions

        anchdb = getAnchors(imdb, ratios, imscale_is_fix, stride, i) # all anchX, anchY, cropped Image, label, numlabel


        # evaluation
        # cls_pred = np.array([(), (), (), ()], dtype=[('prediction', 'f8'), ('x', 'f8'), ('y', 'f8'), ('eval', 'i8')])
        # for anch in anchdb:
        #     pred = predict(anch['image'])
        #     cls_pred['prediction'].append(pred)
        #     cls_pred['x'].append(anch['x'])
        #     cls_pred['y'].append(anch['y'])
        #     cls_pred['eval'].append(pred == anch['vallabel'])
        #
        # print("%f percent of the ima ges are classified correctly", np.sum(cls_pred['eval']))
    # label: end
