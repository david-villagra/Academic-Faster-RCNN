""" configurations for this project

author baiyu
"""
import os
from datetime import datetime
import numpy as np

currentDirectory = os.getcwd()

IMDB_PATH = '/home/tobias/PycharmProjects/Data/images/training'  # Kitti data
LABEL_PATH = '/home/tobias/PycharmProjects/Data/label'           # Kitti labels
TEST_PATH = '/home/tobias/PycharmProjects/Data/images/testing'   # Kitti test
USE_ZERO_MEAN_INPUT_NORMALIZATION = True
KITTILABELS = ["car", "cyclist", "dontcare", "misc", "pedestrian", "person_sitting", "train", "truck", "van"]
RELABEL = {
    "car": "vehicles",
    "cyclist": "vehicles",
    "dontcare": "misc",
    "misc": "misc",
    "pedestrian": "people",
    "person_sitting": "people",
    "train": "vehicles",
    "truck": "vehicles",
    "tram": "vehicles",
    "van": "vehicles",
    "background": "background"
}
CIFARLABELS = ["aquatic mammals", "fish",
                   "flowers",
                   "food containers",
                   "fruit and vegetables",
                   "household electrical devices",
                   "household furniture",
                   "insects",
                   "large carnivores",
                   "large man-made outdoor things",
                   "large natural outdoor scenes",
                   "large omnivores and herbivores",
                   "medium-sized mammals",
                   "non-insect invertebrates",
                   "people",
                   "reptiles",
                   "small mammals",
                   "trees",
                   "vehicles",
                   "background",
                   "misc"]
CIFARLABELS_TO_NUM = dict((name, index) for index,name in enumerate(CIFARLABELS))
NUM_TO_CIFARLABELS = dict((index, name) for index,name in enumerate(CIFARLABELS))
IMG_SCALE = (1242, 375)
GTOVERLAP_CNTR_THRES = 1
GTOVERLAP_AREA_THRES = 4

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR100_TEST_MEAN = (0.3686992156862745, 0.3889972549019608, 0.3773698039215686)
CIFAR100_TEST_STD = (0.20483289760348583, 0.21610958605664488, 0.20964989106753812)

#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 20
MILESTONES = list((np.array(range(6))+1)*5)

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 3

DATA_PATH = currentDirectory+'/cifar-100-python'
WEIGHT_PATH = currentDirectory+'/results/weights'
OUTDIR = currentDirectory+'/results'
SAVE_WEIGHTS = False
USE_ZFNET = 0

ACT = 'relu'  # set manually to 'lrelu'

OPTIM = 'sgd'

LOSS = 'cel'  # cross entropy loss

LR_FCT = 'MSscheduler'


LRDECAY = 0.4

PIXSHUFFLE = False

PIXNOISE = False

DATASHUFFLE = True

SCALE = [32, 64, 96]

NMS = False

WDECAY = 5e-4

LR_INIT = 0.1

