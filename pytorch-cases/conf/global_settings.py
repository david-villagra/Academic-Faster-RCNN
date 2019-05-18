""" configurations for this project

author baiyu
"""
import os
from datetime import datetime
from easydict import EasyDict as edict

cfg = edict()
cfg.IMDB_PATH = '/home/tobias/PycharmProjects/FasterRCNN/venv/images/training'  # Kitti data
cfg.LABEL_PATH = '/home/tobias/PycharmProjects/FasterRCNN/venv/label'           # Kitti labels
cfg.TEST_PATH = '/home/tobias/PycharmProjects/FasterRCNN/venv/images/testing'   # Kitti test
cfg.USE_ZERO_MEAN_INPUT_NORMALIZATION = True
cfg.KITTILABELS = ["car", "cyclist", "dontcare", "misc", "pedestrian", "person_sitting", "train", "truck", "van"]
cfg.RELABEL = {
    "car": "vehicles",
    "cyclist": "vehicles",
    "dontcare": "background",
    "misc": "misc",
    "pedestrian": "people",
    "person_sitting": "people",
    "train": "vehicles",
    "truck": "vehicles",
    "tram": "vehicles",
    "van": "vehicles"
}
cfg.CIFARLABELS = ["aquatic mammals", "fish",
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
cfg.CIFARLABELS_TO_NUM = dict((name, index) for index,name in enumerate(cfg.CIFARLABELS))
cfg.IMG_SCALE = (1242, 375)
cfg.GTOVERLAP_CNTR_THRES = 0.5
cfg.GTOVERLAP_AREA_THRES = 0.5

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
cfg.CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
cfg.CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR100_TEST_MEAN = (0.3686992156862745, 0.3889972549019608, 0.3773698039215686)
CIFAR100_TEST_STD = (0.20483289760348583, 0.21610958605664488, 0.20964989106753812)

#directory to save weights file
cfg.CHECKPOINT_PATH = 'checkpoint'

#total training epoches
cfg.EPOCH = 200
cfg.MILESTONES = [60, 120, 160]

#initial learning rate
#INIT_LR = 0.1

#time of we run the script
cfg.TIME_NOW = datetime.now().isoformat()

#tensorboard log dir
cfg.LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
cfg.SAVE_EPOCH = 10

cfg.DATA_PATH = '/home/tobias/PycharmProjects/PlanB/venv/cifar-100-python'
cfg.WEIGHT_PATH = '/home/tobias/PycharmProjects/PlanB/venv/weights'







