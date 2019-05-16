from easydict import EasyDict as edict

cfg = edict()
cfg.IMDB_PATH = '../../Data/images/training'
cfg.LABLE_PATH = '../../Data/labels'
cfg.TEST_PATH = '../../Data/images/testing'
cfg.USE_ZERO_MEAN_INPUT_NORMALIZATION = True
cfg.LABELS = ['car', 'cyclist', 'dontcare', 'misc', '', '', '', '', '']
cfg.LABELS_KEEP = []