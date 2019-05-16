import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

DEBUG = 1

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes') #error in IDE maybe (in encoding='bytes') because it's pỳthon 3 
    return dict

def show_one_cifar(img, label):
    img_aux = np.reshape(img[:1024], (32,32,1))
    img_aux = np.append(img_aux,np.reshape(img[1024:2048], (32,32,1)),2)
    img_aux = np.append(img_aux,np.reshape(img[2048:], (32,32,1)),2)
    #img_aux = np.append
    #img = np.reshape(img, (32,32))
    print(np.shape(img_aux))
    imgplot = plt.imshow(img_aux)
    plt.show()

def show_subgroup_cifar(imgs, labels):
    print(np.shape(imgs))

class Cifar_data:

    def __init__(self,d_path = ''):
        if(~len(d_path)):
            self.data_path = '/../../CIFAR100'
        self.labels_trn = np.array([])
        self.data_trn = np.array([])
        self.labels_tst = np.array([])
        self.data_tst = np.array([])
    


    def extract_cifar_data(self):
        dirpath = os.getcwd()

        try:
            meta = unpickle(dirpath + self.data_path + '/meta')
            test = unpickle(dirpath + self.data_path + '/test')
            train = unpickle(dirpath + self.data_path + '/train')
            print("CIFAR images loaded")
        except:
            print("Either dataset not downloaded or not in the right folder (CIFAR100)")

        #filenames = [t.decode('utf8') for t in train[b'filenames']]
        self.labels_trn = train[b'coarse_labels']
        self.data_trn = train[b'data']
        self.labels_tst = test[b'coarse_labels'] #there are 20 coarse labels
        self.data_tst = test[b'data']
        #print(self.labels_tst)

        #TODO: extract only the images of the labels that we want, based on the KITTI dataset  


if(DEBUG):
    d = Cifar_data()
    d.extract_cifar_data()
    show_one_cifar(d.data_trn[0,:], d.labels_trn[0])

