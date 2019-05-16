import torch
import numpy as np
from torch.utils.data import Dataset, Dataloader
from config import cfg
from kitti_tools import loadKitti
from roi_tools import getAnchors
from net import predict


if __name__ == '__main__':

    imdb = loadKitti(cfg.IMDB_PATH, cfg.LABEL_PATH) # imdb contains imdb, lbls, positions
    anchdb = getAnchors() # anchdb contains anchors, anch_ctrs, anch_lbls

    # evaluation
    cls_pred = np.array([(), (), (), ()], dtype=[('prediction', 'f8'), ('x', 'f8'), ('y', 'f8'), ('eval', 'i8')])
    for anch in anchdb:
        pred = predict(anch['image'])
        cls_pred['prediction'].append(pred)
        cls_pred['x'].append(anch['x'])
        cls_pred['y'].append(anch['y'])
        cls_pred['eval'].append(pred == anch['vallabel'])

    print("%f percent of the images are classified correctly", np.sum(cls_pred['eval']))