import torch
import numpy as np
from torch.utils.data import Dataset
from conf.global_settings import cfg
from utils.kitti_tools import loadkitti
from utils.roi_tools import getAnchors
import os
from conf import settings
# from net import predict


# if __name__ == '__main__':
def test(ratios=[32, 64, 96], stride=16, imdb_path=settings.IMDB_PATH, label_path=settings.LABEL_PATHF):

    im_files = os.listdir(imdb_path)
    lbl_files = os.listdir(label_path)

    imscale_is_fix = True
    for i in range(len(im_files)):
        imdb = loadkitti(im_files[i], lbl_files[i]) # imdb contains imdb, lbls, positions

        anchdb = getAnchors(imdb, ratios, imscale_is_fix, stride) # all anchX, anchY, cropped Image, label, numlabel

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

