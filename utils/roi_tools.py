import torch
from torch.utils.data import Dataset, Dataloader
from config import cfg

def getAnchors(imdb=None):
    if imdb=None:
        scale = cfg.IMG_SCALE

