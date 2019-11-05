import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.matrix = np.zeros((self.num_class, )*2)

    def accuracy(self):
        Acc = np.diag(self.matrix).sum() / self.matrix.sum()
        return Acc

    def accuracy_class(self):
        Acc = np.diag(self.matrix) / self.matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def precision(self):
        precision = np.diag(self.matrix) / self.matrix.sum(axis=1)
        precision = np.nanmean(precision)
        return precision

    def recall(self):
        recall = np.diag(self.matrix) / self.matrix.sum(axis=0)
        recall = np.nanmean(recall)
        return recall

    def mIOU(self):
        IoU = np.diag(self.matrix) / (np.sum(self.matrix, axis=1) + np.sum(self.matrix, axis=0) - np.diag(self.matrix) + 1e-20)
        mIoU = np.nanmean(IoU)
        return mIoU

    def generate(self, pred, label):
        tmp = pred * self.num_class + label
        self.matrix = np.bincount(tmp, minlength=self.num_class ** 2).reshape((self.num_class, self.num_class))
        return self.mIOU(), self.accuracy_class(), self.precision(), self.recall()


if __name__ == '__main__':
    eval = Evaluator(14)
    pred = np.random.randint(14, size=(10000))
    # label = pred.copy()
    label = np.random.randint(14, size=(10000))
    eval.generate(pred, label)
    print('mIOU    ', eval.mIOU())
    print('acc    ', eval.accuracy())
    print('pre    ', eval.precision())
    print('recall    ', eval.recall())
