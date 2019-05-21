import argparse
from conf import settings
import os
from train import train_net
import numpy as np

def save_results(arguments, results):
    try:
        os.makedirs(arguments.output)
    except OSError:
        pass
    with open(arguments.output + arguments.testname + '.txt', 'w+') as f:
        f.write('net: '+str(arguments.net)+'\n')
        f.write('w: '+str(arguments.w)+'\n')
        f.write('b: '+str(arguments.b)+'\n')
        f.write('base_lr: '+str(arguments.base_lr)+'\n')
        f.write('max_lr: '+str(arguments.max_lr)+'\n')
        f.write('num_iter: '+str(arguments.num_iter)+'\n')
        f.write('use_gpu: '+str(arguments.use_gpu)+'\n')
        f.write('output: '+str(arguments.output)+'\n')
        f.write('optim: '+str(arguments.optim)+'\n')
        f.write('activation: '+str(arguments.act)+'\n')
        f.write('loss: '+str(arguments.loss)+'\n')
        f.write('lr function: '+str(arguments.lr_fct)+'\n')
        f.write('lr initial: '+str(arguments.lr_init)+'\n')
        f.write('lr decay: '+str(arguments.decay)+'\n')
        f.write('pixel shuffling: '+str(arguments.pix_sh)+'\n')
        f.write('pixel noise: '+str(arguments.pix_ns)+'\n')
        f.write('data shuffeling: '+str(arguments.dat_sh)+'\n')
        f.write('scales: '+str(arguments.scales)+'\n')
        f.write('nms: '+str(arguments.nms)+'\n')
        f.write('weight decay: '+str(arguments.wdecay)+'\n')
        f.write('mil: '+str(arguments.mil)+'\n')
        f.write('testname: '+str(arguments.testname)+'\n')
        f.write('warm: '+str(arguments.warm)+'\n')

        f.write("\n")
        f.write("acc ")
        [f.write("%.5f " % i) for i in results[0]]
        f.write("\nloss ")
        [f.write("%.5f " % i) for i in results[1]]
        f.write("\ntest_loss ")
        [f.write("%.5f " % i) for i in results[2]]


# OPTIMIZER  ############################################ DONE ######################
def test_base_setting():
    arg_base = args
    arg_base.testname = 'base'
    results = train_net(arg_base)
    save_results(arg_base, results=results)
    arg_base.testname = ''
    return results


def test_SGD():  ############################################ DONE ######################
    arg_sgd = args
    arg_sgd.optim = 'sgd'
    arg_sgd.testname = 'sgd'
    results = train_net(arg_sgd)
    save_results(arg_sgd, results)
    arg_sgd.optim = 'sgd'
    arg_sgd.testname = ''
    return results


def test_Adam():  ############################################ DONE ######################
    arg_adam = args
    arg_adam.optim = 'adam'
    arg_adam.testname = 'adam'
    results = train_net(arg_adam)
    save_results(arg_adam, results)
    arg_adam.optim = 'sgd'
    arg_adam.testname = ''
    return results


# ACTIVATION
def test_LeakyReLU():  ############################################ DONE ######################
    arg_lrelu = args
    arg_lrelu.act = 'lrelu'
    settings.ACT = 'lrelu'
    arg_lrelu.testname = 'lrelu'
    results = train_net(arg_lrelu)
    save_results(arg_lrelu, results)
    arg_lrelu.act = 'relu'
    settings.ACT = 'relu'
    arg_lrelu.testname = ''
    return results


def test_ReLU():  ############################################ DONE ######################
    arg_relu = args
    arg_relu.act = 'relu'
    arg_relu.testname = 'relu'
    results = train_net(arg_relu)
    save_results(arg_relu, results)
    return results

# LOSS
# def test_SmoothL1():   ############################################ DONE ######################
#     arg_smoothl1 = args
#     arg_smoothl1.loss = 'smoothl1'
#     arg_smoothl1.testname = 'smoothl1'
#     results = train_net(arg_smoothl1)
#     save_results(arg_smoothl1, results)
#     return results


def test_crossentropyLoss():  ############################################ DONE ######################
    arg_crentropy = args
    arg_crentropy.loss = 'cel'
    arg_crentropy.testname = 'cel'
    results = train_net(arg_crentropy)
    save_results(arg_crentropy, results)
    return results


def test_MultiLabelMarginLoss():  ############################################ DONE ######################
    arg_MLML = args
    arg_MLML.loss = 'mlml'
    arg_MLML.testname = 'mlml'
    results = train_net(arg_MLML)
    save_results(arg_MLML, results)
    arg_MLML.loss = 'cel'
    arg_MLML.testname = ''
    return results


# BATCHSIZE
def test_batchisze(size):   ############################################ DONE ######################
    arg_batch = args
    arg_batch.b = size
    arg_batch.testname = 'batch'+str(size)
    results = train_net(arg_batch)
    save_results(arg_batch, results)
    arg_batch.b = 100
    arg_batch.testname = ''
    return results


# ROI
def test_scales(scales):   ############################################ DONE ######################
    arg_scale = args
    arg_scale.scales = scales
    string = ""
    string = [str(i)+" " for i in scales]
    arg_scale.testname = 'scales'+ string
    results = train_net(arg_scale)
    save_results(arg_scale, results)
    arg_scale.scales = settings.SCALE
    arg_scale.testname = ''
    return results


def test_nms(): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NOT DONE
    arg_nms = args
    arg_nms.nms = True
    arg_nms.testname = 'nms'
    results = train_net(arg_nms)
    save_results(arg_nms, results)
    arg_nms.nms = False
    arg_nms.testname = ''
    return results


# LEARNING RATE
def test_cyclicLearning():    ############################################ DONE ######################
    arg_cyc = args
    arg_cyc.lr_fct = 'cyclic'
    arg_cyc.testname = 'cyclic'
    results = train_net(arg_cyc)
    save_results(arg_cyc, results)
    arg_cyc.lr_fct = 'MSscheduler'
    arg_cyc.testname = ''
    return results


def test_LRScheduler():  ############################################ DONE ######################
    arg_sched = args
    arg_sched.lr_fct = 'MSscheduler'
    arg_sched.testname = 'MSscheduler'
    results = train_net(arg_sched)
    save_results(arg_sched, results)
    return results


def test_decay(value):    ############################################ DONE ######################
    arg_dec = args
    arg_dec.decay = value
    arg_dec.testname = 'decay' +str(value)
    results = train_net(arg_dec)
    save_results(arg_dec, results)
    arg_dec.decay = settings.LRDECAY
    arg_dec.testname = ''
    return results


def test_milestone(list,decay):   ############################################ DONE ######################
    arg_mil = args
    arg_mil.mil = list
    arg_mil.decay = decay
    string = ""
    string = [string +" "+ str(i) for i in list]
    arg_mil.testname = 'milestone_' +str(string)  + '_decay_' +str(decay)
    results = train_net(arg_mil)
    save_results(arg_mil, results)
    arg_mil.mil = settings.MILESTONES
    arg_mil.decay = settings.LRDECAY
    arg_mil.testname = ''
    return results


def test_pixelShuffle():  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! not done yet'
    arg_ps = args
    arg_ps.pix_sh = True
    arg_ps.testname = 'pixelshuffle'
    results = train_net(arg_ps)
    save_results(arg_ps, results)
    arg_ps.pix_sh = False
    arg_ps.testname = ''
    return results


def test_pixelNoise():
    arg_pns = args
    arg_pns.pix_ns = True
    arg_pns.testname = 'pixelNoise'
    results = train_net(arg_pns)
    save_results(arg_pns, results)
    arg_pns.pix_ns = False
    arg_pns.testname = ''
    return results


def test_noDataShuffle(): ############################################ DONE ######################
    arg_ds = args
    arg_ds.data_sh = False
    arg_ds.testname = 'nodatashuffle'
    results = train_net(arg_ds)
    save_results(arg_ds, results)
    arg_ds.data_sh = True
    arg_ds.testname = ''
    return results


def test_weightdecay(value):  ############################################ DONE ######################
    arg_wd = args
    arg_wd.wdecay = value
    arg_wd.testname = 'weightdecay'
    results = train_net(arg_wd)
    save_results(arg_wd, results)
    arg_wd.wdecay = settings.WDECAY
    arg_wd.testname = ''
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='googlenet', help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=100, help='batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('-max_lr', type=float, default=0.1, help='max learning rate')
    parser.add_argument('-num_iter', type=int, default=settings.EPOCH, help='num of iteration')
    parser.add_argument('-use_gpu', nargs='+', type=bool, default=True, help='gpu device')
    parser.add_argument('-output', type=str, default=settings.OUTDIR, help='output directory')
    parser.add_argument('-optim', type=str, default=settings.OPTIM, help='optimizer to use')
    parser.add_argument('-act', type=str, default=settings.ACT, help='activation fct')
    parser.add_argument('-loss', type=str, default=settings.LOSS, help='loss fct')
    parser.add_argument('-lr_fct', type=str, default=settings.LR_FCT, help='LR scheduler or cyclic')
    parser.add_argument('-lr_init', type=float, default=settings.LR_INIT, help='LR initial value')
    parser.add_argument('-decay', type=float, default=settings.LRDECAY, help='LR decay')
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
    else:
        settings.USE_ZFNET = 0

    #     res_sgd = test_base_setting() # equal to test_ReLU#  #We already performed this test

    # res_sgd = test_SGD() # redundant to the first case

    #     res_adam =test_Adam()  # uncommented because it performed already

    # res_smoothl1 = test_SmoothL1() # doesnot work for some reason

    # res_crossentr = test_crossentropyLoss()  #redundant to base case

    # res_MLML = test_MultiLabelMarginLoss() # removed to reduce test amount

    # res_batch1 = test_batchsize(32)   # removed to reduce test amount (wo don't havea high range for testing batch sizes

    # res_batch1 = test_batchsize(64)  # removed since it will kill the run

    #     res_cycl = test_cyclicLearning()  # or fixed rate

    # res_LRs = test_LRScheduler() #

    #     res_mil1 = test_milestone(list((np.array(range(np.int8(settings.EPOCH/2)))+1)*2),0.8)
    
    #     res_mil2 = test_milestone(list((np.array(range(np.int8(settings.EPOCH / 2))) + 1) * 2), 0.4)
   
    #     res_mil3 = test_milestone(list((np.array(range(np.int8(settings.EPOCH/4)))+1)*4), 0.4)  # removed to reduce test amount
   
    # res_mil4 = test_milestone(list((np.array(range(np.int8(settings.EPOCH/8))) + 1) * 8), 0.5)  # reduced to reduce test amount
    
    # res_pixns = test_pixelNoise()
    
    # res_datsh = test_noDataShuffle()  #removed to reduce test amount
    
    #     res_lrelu = test_LeakyReLU()
    args.num_iter = 20
    args.wdecay = 0.4
    args.mil = range(19)+1
    gamma = [0.2, 0.4, 0.6, 0.8]
    for g in gamma:
        res = test_decay(g)

    args.mil = [2, 4, 8, 16]
    res = test_decay(0.75)

    args.mil = range(19)+1
    args.decay = 0.4
    weightdecay = gamma
    for w in weightdecay:
	res = test_weightdecay(w)

    args.decay = 0.4 
    args.wdecay = 0.4
    args.num_iter = 40
    args.lr_fct = 'cyclic'
    res = test_cyclicLearning()  

    batch = [20 50 100 200]
    args.lr_fct = "MSscheduler"
    for b in batch:
        res = test_batchsize(b)
