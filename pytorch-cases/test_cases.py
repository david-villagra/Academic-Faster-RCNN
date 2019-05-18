import unittest
from easydict import EasyDict as edict
import argparse
from config import cfg

configs = edict()


class TestOptimizer(unittest.TestCase):

    def test_SGD(self):
        self.assertEqual(True, False)

    def test_Adam(self):
        self.assertEqual(True, False)

    def test_RMSprop(self):
        self.assertEqual(True, False)

    def test_RReLU(self): # or SELU or
        self.assertEqual(True, False)


class TestActivation(unittest.TestCase):

    def test_ELU(self):
        self.assertEqual(True, False)

    def test_LeakyReLU(self):
        self.assertEqual(True, False)

    def test_ReLU(self):
        self.assertEqual(True, False)

    def test_RReLU(self): # or SELU or
        self.assertEqual(True, False)


class TestPooling(unittest.TestCase):

    def test_maxPool(self):
        self.assertEqual(True, False)

    def test_avgPool(self):
        self.assertEqual(True, False)

    def test_LPPool(self):
        self.assertEqual(True, False)


class TestNormalization(unittest.TestCase):

    def test_batchsize(self):
        self.assertEqual(True, False)


class TestLossFunction(unittest.TestCase):

    def test_SmoothL1(self):
        self.assertEqual(True, False)

    def test_crossentropyLoss(self):
        self.assertEqual(True, False)

    def test_MultiLabelMarginLoss(self):
        self.assertEqual(True, False)

class TestLayer(unittest.TestCase):

    def test_layer1(self):
        self.assertEqual(True, False)
    # ...

    def test_droput(self):
        self.assertEqual(True, False)


class TestROI(unittest.TestCase):

    def test_scales(self):
        self.assertEqual(True, False)

    def test_nms(self):
        self.assertEqual(True, False)


class TestNetParameters(unittest.TestCase):

    def test_cyclicLearning(self):
        self.assertEqual(True, False)

    def test_LRScheduler(self):         ################################ LR_finder implemented, use it ################
        self.assertEqual(True, False)

    def test_decay(self):
        self.assertEqual(True, False)

    def test_pixelShuffle(self):
        self.assertEqual(True, False)

    def test_pixelNoise(self):
        self.assertEqual(True, False)

    def test_dataShuffle(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-w', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-base_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('-max_lr', type=float, default=10, help='max learning rate')
    parser.add_argument('-num_iter', type=int, default=100, help='num of iteration')
    parser.add_argument('-gpus', nargs='+', type=int, default=0, help='gpu device')
    parser.add_argument('-output', type=str, default=cfg.OUTDIR, help='output directory')
    args = parser.parse_args()


    unittest.main()

