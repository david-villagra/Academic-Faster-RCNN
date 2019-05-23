import argparse
from conf import settings
import os
from test import test_one
import numpy as np

if __name__ == '__main__':

    if settings.TEST_WITH_RANDOM_BACKGROUND is False:
        name_img = 'testanchor1024'
    else:
        name_img = 'background_test4'
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='googlenet', help='net type')
    parser.add_argument('-path_pth', type=str, default=dir_path+'/../DL_LargeFiles/googlenet-50-regular.pth', help='Path to trained model')
    parser.add_argument('-path_img', type=str, default=dir_path+'/images/' + name_img + '.jpg', help='Path to image file')
    parser.add_argument('-use_gpu', nargs='+', type=bool, default=False, help='gpu device')
    args = parser.parse_args()

    if args.net is 'zfnet':
        settings.USE_ZFNET = 1
    else: 
        settings.USE_ZFNET = 0
        
    settings.WEIGHT_PATH = dir_path + args.path_pth

    result = test_one(args, name_img)