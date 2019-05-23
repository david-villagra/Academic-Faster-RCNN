# -*- coding: utf-8 -*-
import codecs
import os
import pickle
#import io
#from PIL import Image
import numpy as np


def im_to_cifar(image_array, class_im):

    # Input: 3D array (32 x 32 x 3)

    im_array_R = image_array[:, :, 0]
    im_array_G = image_array[:, :, 1]
    im_array_B = image_array[:, :, 2]

    byte_array = np.array(list(class_im) + list(im_array_R) + list(im_array_G) + list(im_array_B))

    print(byte_array) # Wanted to check, but it doesnt print anything...
    return byte_array


def randomBackground():
    i = np.random.randint(2, size=1)
    weights = np.random.uniform(0, 1, size=(3,))
    if i == 0:
        image = np.random.uniform(0, 255, (32*32*3,))
        image[0:32*32] = weights[0] * image[0:32*32]
        image[32 * 32:2 * 32 * 32] = weights[0] * image[32 * 32:2*32*32]
        image[2 * 32 * 32:3 * 32 * 32] = weights[0] * image[2 * 32 * 32:3 * 32 * 32]
        image = np.uint8(image)
    elif i == 1:
        image = np.random.beta(np.random.uniform(0, 1, 1), np.random.uniform(0, 1, 1), (32*32*3,))*255
        image[0:32*32] = weights[0] * image[0:32*32]
        image[32 * 32:2 * 32 * 32] = weights[0] * image[32 * 32:2*32*32]
        image[2 * 32 * 32:3 * 32 * 32] = weights[0] * image[2 * 32 * 32:3 * 32 * 32]
        image = np.uint8(image)
    return image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def dopickle(dict, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict, fo)

def read_cifar(what):
    meta_path = '/home/tobias/PycharmProjects/PlanB/venv/cifar-100-python/meta'
    test_path = '/home/tobias/PycharmProjects/PlanB/venv/cifar-100-python/test'
    train_path = '/home/tobias/PycharmProjects/PlanB/venv/cifar-100-python/train'

    if what is 'meta':
        data_path = meta_path
        """
            'fine_label_names'
            'coarse_label_names'
        """
    elif what is 'test':
        data_path = test_path
    elif what is 'train':
        data_path = train_path
        """ b'filenames'
            b'fine_labels'
            b'data'
            b'coarse_labels'
            b'batch_label'
            ''
        """

    dict = unpickle(data_path)
    return dict


def generate_new_cifar():
    # meta_dict = read_cifar('meta')
    # meta_dict[b'fine_label_names'].append("b'background'")
    # meta_dict[b'coarse_label_names'].append("b'background'")
    folder = '/home/tobias/PycharmProjects/PlanB/venv/cifar-100-python/'
    # dopickle(meta_dict, folder + 'meta_new')
    # for i in range(2500):
    #
    #     train_dict = read_cifar('train')
    #     filename = str(i)
    #     filename = filename.zfill(6)
    #     filename = "b'background_s_" + filename + ".png"
    #     train_dict[b'filenames'].append(filename)
    #     train_dict[b'fine_labels'].append(20)
    #     train_dict[b'coarse_labels'].append(100)
    #     train_dict[b'data'] = np.append(train_dict[b'data'], randomBackground())
    # dopickle(train_dict, folder + 'train_new')

    for i in range(500):
        test_dict = read_cifar('test')
        filename = str(i+2500)
        filename = filename.zfill(6)
        filename = "b'background_s_" + filename + ".png"
        test_dict[b'filenames'].append(filename)
        test_dict[b'fine_labels'].append(20)
        test_dict[b'coarse_labels'].append(100)
        test_dict[b'data'] = np.append(test_dict[b'data'], randomBackground())
    dopickle(test_dict, folder + 'test_new')
