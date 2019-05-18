import unittest
from easydict import EasyDict as edict
import argparse
import config as cfg


# OPTIMIZER
def test_SGD():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

def test_Adam():
    arg_adam = args
    arg_adam.optim = 'adam'
    results = train_net(arg_adam)
    return results

# ACTIVATION
def test_LeakyReLU():
    arg_lrelu = args
    arg_lrelu.act = 'sgd'
    results = train_net(arg_lrelu)
    return results

def test_ReLU():
    arg_relu = args
    arg_relu.act = 'sgd'
    results = train_net(arg_relu)
    return results

# LOSS
def test_SmoothL1():
    arg_smoothl1 = args
    arg_smoothl1.loss = 'sgd'
    results = train_net(arg_smoothl1)
    return results

def test_crossentropyLoss():
    arg_crentropy = args
    arg_crentropy.loss = 'sgd'
    results = train_net(arg_crentropy)
    return results

def test_MultiLabelMarginLoss():
    arg_MLML = args
    arg_MLML.loss = 'sgd'
    results = train_net(arg_MLML)
    return results

# BATCHSIZE
def test_batchisze(size):
    arg_batch = args
    arg_batch.b = size
    results = train_net(arg_batch)
    return results

# ROI
def test_scales(scales):
    arg_scale = args
    arg_scale.optim = 'sgd'
    results = train_net(arg_scale)
    return results

def test_nms():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

# LEARNING RATE
def test_cyclicLearning():  # or fixed rate
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

def test_LRScheduler():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

def test_decay():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

def test_pixelShuffle():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

def test_pixelNoise():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results

def test_dataShuffle():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    results = train_net(arg_sgd)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('-max_lr', type=float, default=10, help='max learning rate')
    parser.add_argument('-num_iter', type=int, default=100, help='num of iteration')
    parser.add_argument('-use_gpu', nargs='+', type=int, default=0, help='gpu device')
    parser.add_argument('-output', type=str, default=cfg.OUTDIR, help='output directory')
    parser.add_argument('-optim', type=str, default=cfg.OPTIM, help='optimizer to use')
    parser.add_argument('-act', type=str, default=cfg.ACT, help='activation fct')
    parser.add_argument('-loss', type=str, default=cfg.LOSS, help='loss fct')
    parser.add_argument('-lr_fct', type=str, default=cfg.LR, help='LR fct')
    parser.add_argument('-decay', type=str, default=cfg.DECAY, help='LR decay')
    parser.add_argument('-pix_sh', type=bool, default=cfg.PIXSHUFFLE, help='')
    parser.add_argument('-pix_ns', type=bool, default=cfg.PIXNOISE, help='')
    parser.add_argument('-dat_sh', type=bool, default=cfg.DATASHUFFLE, help='')
    parser.add_argument('-scales', type=int, default=cfg.SCALE, help='')
    parser.add_argument('-nms', type=bool, default=cfg.nms, help='')
    args = parser.parse_args()

    if parser.net is 'zfnet':
        cfg.USE_ZFNET = 1


    res_sgd = test_SGD()
    res_adam =test_Adam()
    res_lrelu = test_LeakyReLU()
    res_relu = test_ReLU()
    res_smoothl1 = test_SmoothL1()
    res_crossentr = test_crossentropyLoss()
    res_MLML = test_MultiLabelMarginLoss()
    res_batch1 = test_batchisze(32)
    res_batch1 = test_batchisze(64)
    # test_scales()
    res_cycl = test_cyclicLearning()  # or fixed rate
    res_LRs = test_LRScheduler() ################################ LR_finder implemented, use it ################
    res_decay1 = test_decay(0.2)
    res_decay2 = test_decay(0.1)
    res_pixsh = test_pixelShuffle()
    res_pixns = test_pixelNoise()
    res_datsh = test_dataShuffle()
