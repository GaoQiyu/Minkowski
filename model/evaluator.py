import numpy as np
import math


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.matrix = np.zeros((self.num_class, )*2)

    def mIOU(self, pred, label):
        tmp = pred*self.num_class + label
        self.matrix = np.bincount(tmp, minlength=self.num_class**2).reshape((self.num_class, self.num_class))
        IoU = np.diag(self.matrix) / (np.sum(self.matrix, axis=1) + np.sum(self.matrix, axis=0) - np.diag(self.matrix) + 1e-10)
        mIoU = np.nanmean(IoU)
        return IoU, mIoU