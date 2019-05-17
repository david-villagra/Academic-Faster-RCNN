from easydict import EasyDict as edict

cfg = edict()
cfg.IMDB_PATH = '/home/tobias/PycharmProjects/FasterRCNN/venv/images/training'
cfg.LABLE_PATH = '/home/tobias/PycharmProjects/FasterRCNN/venv/label'
cfg.TEST_PATH = '/home/tobias/PycharmProjects/FasterRCNN/venv/images/testing'
cfg.USE_ZERO_MEAN_INPUT_NORMALIZATION = True
cfg.KITTILABELS = ["car", "cyclist", "dontcare", "misc", "pedestrian", "person_sitting", "train", "truck", "van"]
cfg.RELABEL = {
    "car": "vehicle",
    "cyclist": "vehicle",
    "dontcare": "background",
    "misc": "misc",
    "pedestrian": "people",
    "person_sitting": "people",
    "train": "vehicle",
    "truck": "vehicle",
    "van": "vehicle"
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
                   "vehicles"]
cfg.CIFARLABELS_TO_NUM = dict((name, index) for index,name in enumerate(cfg.CIFARLABELS))
cfg.IMG_SCALE = (1242, 375)
cfg.GTOVERLAP_THRES = 0.5