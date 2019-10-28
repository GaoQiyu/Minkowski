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


if __name__ == '__main__':
    eval = Evaluator(14)
    pred = np.random.randint(14, size=(10000))
    label = pred.copy()
    Iou, mIoU = eval.mIOU(pred, label)
    print(Iou)
    print(mIoU)