import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.matrix = np.zeros((self.num_class, )*2)

    def Accuracy(self):
        Acc = np.diag(self.matrix).sum() / self.matrix.sum()
        return Acc

    def precision(self):
        precision = np.diag(self.matrix) / self.matrix.sum(axis=1)
        precision = np.nanmean(self.remove_zero(precision))
        return precision

    def recall(self):
        recall = np.diag(self.matrix) / self.matrix.sum(axis=0)
        recall = np.nanmean(self.remove_zero(recall))
        return recall

    def mIOU(self):
        IoU = np.diag(self.matrix) / (np.sum(self.matrix, axis=1) + np.sum(self.matrix, axis=0) - np.diag(self.matrix) + 1e-20)
        mIoU = np.nanmean(self.remove_zero(IoU))
        return IoU, mIoU

    def generate(self, pred, label):
        tmp = pred * self.num_class + label
        self.matrix = np.bincount(tmp, minlength=self.num_class ** 2).reshape((self.num_class, self.num_class))
        _, mIOU = self.mIOU()
        Acc = self.Accuracy()
        precision = self.precision()
        recall = self.recall()
        return mIOU, Acc, precision, recall

    def remove_zero(self, data):
        tmp = []
        for i in data:
            if i != 0 and i != np.nan:
                tmp.append(i)
        return np.array(tmp)


if __name__ == '__main__':
    eval = Evaluator(14)
    pred = np.random.randint(14, size=(10000))
    # label = pred.copy()
    label = np.random.randint(14, size=(10000))
    eval.generate(pred, label)
    IOU, mIOU = eval.mIOU()
    print('mIOU    ', IOU)
    print('mIOU    ', mIOU)
    print('Acc    ', eval.Accuracy())
    print('pre    ', eval.precision())
    print('recall    ', eval.recall())
