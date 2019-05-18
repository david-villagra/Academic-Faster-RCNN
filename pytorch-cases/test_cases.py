import unittest
from easydict import EasyDict as edict
import argparse
from conf import settings
import os
from train import train_net


def save_results(arguments, results):
    try:
        os.makedirs(arguments.output)
    except OSError:
        pass
    with open(arguments.output + arguments.testname + '.txt', 'w+') as f:
        f.write('net: '+str(arguments.net))
        f.write('w: '+str(arguments.w))
        f.write('b: '+str(arguments.b))
        f.write('base_lr: '+str(arguments.base_lr))
        f.write('max_lr: '+str(arguments.max_lr))
        f.write('num_iter: '+str(arguments.num_iter))
        f.write('use_gpu: '+str(arguments.use_gpu))
        f.write('output: '+str(arguments.output))
        f.write('optim: '+str(arguments.optim))
        f.write('activation: '+str(arguments.act))
        f.write('loss: '+str(arguments.loss))
        f.write('lr function: '+str(arguments.lr_fct))
        f.write('lr initial: '+str(arguments.lr_init))
        f.write('lr decay: '+str(arguments.decay))
        f.write('pixel shuffling: '+str(arguments.pix_sh))
        f.write('pixel noise: '+str(arguments.pix_ns))
        f.write('data shuffeling: '+str(arguments.dat_sh))
        f.write('scales: '+str(arguments.scales))
        f.write('nms: '+str(arguments.nms))
        f.write('weight decay: '+str(arguments.wdecay))
        f.write('mil: '+str(arguments.mil))
        f.write('testname: '+str(arguments.testname))
        f.write('warm: '+str(arguments.warm))

        f.write("\n")
        f.write("acc\n")
        [f.write("%d " % i) for i in results[0]]
        f.write("loss\n")
        [f.write("%d " % i) for i in results[1]]
        f.write("test_loss\n")
        [f.write("%d " % i) for i in results[2]]


# OPTIMIZER
def test_base_setting():
    arg_base = args
    arg_base.testname = 'base'
    results = train_net(arg_base)
    save_results(arg_base, results=results)
    return results


def test_SGD():
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    arg_sgd.testname = 'sgd'
    results = train_net(arg_sgd)
    save_results(arg_sgd, results)
    return results


def test_Adam():
    arg_adam = args
    arg_adam.optim = 'adam'
    arg_adam.testname = 'adam'
    results = train_net(arg_adam)
    save_results(arg_adam, results)
    return results


# ACTIVATION
def test_LeakyReLU():
    arg_lrelu = args
    arg_lrelu.act = 'lrelu'
    arg_lrelu.testname = 'lrelu'
    results = train_net(arg_lrelu)
    save_results(arg_lrelu, results)
    return results


def test_ReLU():
    arg_relu = args
    arg_relu.act = 'relu'
    arg_relu.testname = 'relu'
    results = train_net(arg_relu)
    save_results(arg_relu, results)
    return results

# LOSS
def test_SmoothL1():
    arg_smoothl1 = args
    arg_smoothl1.loss = 'smoothl1'
    arg_smoothl1.testname = 'smoothl1'
    results = train_net(arg_smoothl1)
    save_results(arg_smoothl1, results)
    return results


def test_crossentropyLoss():
    arg_crentropy = args
    arg_crentropy.loss = 'crossentropy'
    arg_crentropy.testname = 'crossentropy'
    results = train_net(arg_crentropy)
    save_results(arg_crentropy, results)
    return results


def test_MultiLabelMarginLoss():
    arg_MLML = args
    arg_MLML.loss = 'mlml'
    arg_MLML.testname = 'mlml'
    results = train_net(arg_MLML)
    save_results(arg_MLML, results)
    return results


# BATCHSIZE
def test_batchisze(size):
    arg_batch = args
    arg_batch.b = size
    arg_batch.testname = 'batch'+str(size)
    results = train_net(arg_batch)
    save_results(arg_batch, results)
    return results


# ROI
def test_scales(scales):
    arg_scale = args
    arg_scale.scales = scales
    string = [str(i)+" " for i in scales]
    arg_scales.testname = 'scales'+ string
    results = train_net(arg_scale)
    save_results(arg_scale, results)
    return results


def test_nms():
    arg_nms = args
    arg_nms.nms = True
    arg_nms.testname = 'nms'
    results = train_net(arg_nms)
    save_results(arg_nms, results)
    return results


# LEARNING RATE
def test_cyclicLearning():  # or fixed rate
    arg_cyc = args
    arg_cyc.lr_fct = 'cyclic'
    arg_cyc.testname = 'cyclic'
    results = train_net(arg_cyc)
    save_results(arg_cyc, results)
    return results


def test_LRScheduler():
    arg_sched = args
    arg_sched.lr_fct = 'scheduler'
    arg_sched.testname = 'scheduler'
    results = train_net(arg_sched)
    save_results(arg_sched, results)
    return results


def test_decay(value):
    arg_dec = args
    arg_dec.decay = value
    arg_dec.testname = 'decay' +str(value)
    results = train_net(arg_dec)
    save_results(arg_dec, results)
    return results


def test_milestone(list,decay):
    arg_mil = args
    arg_mil.milestone = list
    arg_mil.decay = decay
    string = [str(i)+" " for i in list]
    arg_mil.testname = 'milestone_' +string  + '_decay_' +str(value)
    results = train_net(arg_mil)
    save_results(arg_mil, results)
    return results


def test_pixelShuffle():
    arg_ps = args
    arg_ps.pix_sh = True
    arg_ps.testname = 'pixelshuffle'
    results = train_net(arg_ps)
    save_results(arg_ps, results)
    return results


def test_pixelNoise():
    arg_pns = args
    arg_pns.pix_ns = True
    arg_pns.testname = 'pixelNoise'
    results = train_net(arg_pns)
    save_results(arg_pns, results)
    return results


def test_dataShuffle():
    arg_ds = args
    arg_ds.data_sh = True
    arg_ds.testname = 'datashuffle'
    results = train_net(arg_ds)
    save_results(arg_pns, results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='zfnet', help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=5000, help='batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('-max_lr', type=float, default=10, help='max learning rate')
    parser.add_argument('-num_iter', type=int, default=settings.EPOCH, help='num of iteration')
    parser.add_argument('-use_gpu', nargs='+', type=bool, default=False, help='gpu device')
    parser.add_argument('-output', type=str, default=settings.OUTDIR, help='output directory')
    parser.add_argument('-optim', type=str, default=settings.OPTIM, help='optimizer to use')
    parser.add_argument('-act', type=str, default=settings.ACT, help='activation fct')
    parser.add_argument('-loss', type=str, default=settings.LOSS, help='loss fct')
    parser.add_argument('-lr_fct', type=str, default=settings.LR_FCT, help='LR scheduler or cyclic')
    parser.add_argument('-lr_init', type=str, default=settings.LR_INIT, help='LR initial value')
    parser.add_argument('-decay', type=str, default=settings.LRDECAY, help='LR decay')
    parser.add_argument('-pix_sh', type=bool, default=settings.PIXSHUFFLE, help='')
    parser.add_argument('-pix_ns', type=bool, default=settings.PIXNOISE, help='')
    parser.add_argument('-dat_sh', type=bool, default=settings.DATASHUFFLE, help='')
    parser.add_argument('-scales', type=int, default=settings.SCALE, help='')
    parser.add_argument('-nms', type=bool, default=settings.NMS, help='')
    parser.add_argument('-wdecay', type=bool, default=settings.WDECAY, help='')
    parser.add_argument('-mil', type=int, default=settings.MILESTONES, help='')
    parser.add_argument('-testname', type=str, required=False)
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    args = parser.parse_args()

    if args.net is 'zfnet':
        settings.USE_ZFNET = 1

    res_sgd = test_base_setting()
    # res_sgd = test_SGD()
    # res_adam =test_Adam()
    # res_lrelu = test_LeakyReLU()
    # res_relu = test_ReLU()
    # res_smoothl1 = test_SmoothL1()
    # res_crossentr = test_crossentropyLoss()
    # res_MLML = test_MultiLabelMarginLoss()
    # res_batch1 = test_batchisze(32)
    # res_batch1 = test_batchisze(64)
    # # test_scales()
    # res_cycl = test_cyclicLearning()  # or fixed rate
    # res_LRs = test_LRScheduler() #
    # res_mil1 = test_milestone(list((np.array(range(EPOCH/2))+1)*2),0.8)
    # res_mil2 = test_milestone(list((np.array(range(EPOCH / 2)) + 1) * 2), 0.4)
    # res_mil3 = test_milestone(list((np.array(range(EPOCH/4))+1)*4), 0.5)
    # res_mil4 = test_milestone(list((np.array(range(EPOCH/8)) + 1) * 8), 0.5)
    # #res_decay1 = test_decay(0.8)
    # #res_decay2 = test_decay(0.2)
    # res_pixsh = test_pixelShuffle()
    # res_pixns = test_pixelNoise()
    # res_datsh = test_dataShuffle()
