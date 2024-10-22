import os
import torch
import math
import time
import numpy as np
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
import model.res16unet as ResUNet
from model.lr_scheduler import PolyLR
from model.evaluator import Evaluator
from data.dataset import initialize_data_loader
from data.S3DIS import S3DISDataset


class Trainer(object):
    def __init__(self, config_):
        self.config = config_
        self.best_pred = -math.inf
        self.train_iter_number = 0
        self.val_iter_number = 0
        self.epoch = 0
        self.class_name = self.config["class_label"]
        if self.config["multi_gpu"]:
            self.device_list = list(range(torch.cuda.device_count()))
            self.device = self.device_list[0]
        else:
            self.device = torch.device('cuda')
        self.loss_value = torch.tensor(0.0, requires_grad=True).to(self.device)
        self.point_number = self.config["point_num"]
        self.batch_size = self.config["batch_size"]
        self.model = ResUNet.Res16UNet34C(3, self.config["class"])
        if self.config["fine_tune"]:
            model_dict = torch.load(os.path.join(self.config["resume_path"], 'weights_14.pth'), map_location=lambda storage, loc: storage.cuda(self.device))
            self.model.load_state_dict(model_dict)
            self.optimizer = torch.optim.SGD([
                {'params': self.model.convtr7p2s2.parameters(), 'lr': self.config["lr"] / 1e2},
                {'params': self.model.bntr7.parameters(), 'lr': self.config["lr"] / 1e2},
                {'params': self.model.block8.parameters(), 'lr': self.config["lr"] / 1e1},
                {'params': self.model.final.parameters(), 'lr': self.config["lr"]}],
                lr=self.config["lr"] / 1e4, momentum=self.config["momentum"], weight_decay=1e-4)
        if self.config["use_cuda"]:
            self.model = self.model.to(self.device)
        if self.config["multi_gpu"]:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_list)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.config['ignore_label'])

        self.train_data = initialize_data_loader(S3DISDataset, self.config, phase='TRAIN', threads=1, augment_data=True, shuffle=True, repeat=True, batch_size=1, limit_numpoints=False)
        self.val_data = initialize_data_loader(S3DISDataset, self.config, threads=1, phase='VAL', augment_data=False, shuffle=True, repeat=False, batch_size=1, limit_numpoints=False)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'], momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        self.lr_scheduler = PolyLR(self.optimizer, max_iter=60000, power=self.config['poly_power'], last_step=-1)

        log_path = os.path.join(self.config["log_path"], str(time.time()))
        os.mkdir(log_path) if not os.path.exists(log_path) else None
        self.summary = SummaryWriter(log_path)
        self.evaluator = Evaluator(self.config["class"])

        self.load()

    def train(self, epoch_):
        epoch_loss = []
        self.model.train()
        for ith, data_dict in enumerate(self.train_data):
            point, labels = self.data_preprocess(data_dict, 'train')
            output_sparse = self.model(point)
            pred = output_sparse.F
            self.reset_loss(labels)
            self.loss_value = self.loss(pred, labels) + self.loss_value
            epoch_loss.append(self.loss(pred, labels).item() / self.config["accumulate_gradient"])
            self.train_iter_number += 1
            if self.train_iter_number % self.batch_size == 0:
                self.loss_value /= (self.config["accumulate_gradient"]*self.batch_size)
                self.loss_value.backward()
                lr_value = self.optimizer.state_dict()['param_groups'][0]['lr']
                self.optimizer.step()
                self.lr_scheduler.step(self.train_iter_number)
                self.optimizer.zero_grad()
                torch.cuda.empty_cache()
                self.summary.add_scalar('train/loss: ', self.loss_value, self.train_iter_number)
                self.summary.add_scalar('train/lr: ', lr_value, self.train_iter_number)
                print("train epoch:  {}/{}, ith:  {}/{}, loss:  {:.4f}, lr:  {:.6f}".format(epoch_, self.config['epoch'], self.train_iter_number, len(self.train_data), self.loss_value.item(), lr_value))
                self.loss_value = 0
            if ith == len(self.train_data)-1:
                break
        average_loss = np.nanmean(epoch_loss)
        self.summary.add_scalar('train/loss_epoch: ', average_loss, epoch_)
        print("epoch:    {}/{}, average_loss:    {:.4f}".format(epoch_, self.config['epoch'], average_loss))
        print('------------------------------------------------------------------')

    def eval(self, epoch_):
        self.model.eval()
        torch.cuda.empty_cache()
        IOU_epoch = []
        mIOU_epoch = []
        Acc_epoch = []
        precision_epoch = []
        recall_epoch = []
        epoch_loss = []
        for ith, data_dict in enumerate(self.val_data):
            point, labels = self.data_preprocess(data_dict, 'val')
            with torch.no_grad():
                output = self.model(point)
            pred = output.F
            loss_eval = self.loss(pred, labels) / self.config["accumulate_gradient"]
            epoch_loss.append(loss_eval.item())
            mIOU, Acc, precision, recall = self.evaluator.generate(pred.max(1)[1].cpu(), labels.cpu())
            IOU, _ = self.evaluator.mIOU()

            IOU_epoch.append(IOU)
            mIOU_epoch.append(mIOU)
            Acc_epoch.append(Acc)
            precision_epoch.append(precision)
            recall_epoch.append(recall)

            self.val_iter_number += 1
            self.summary.add_scalar('val/mIOU', mIOU, self.val_iter_number)
            self.summary.add_scalar('val/accuracy', Acc, self.val_iter_number)
            self.summary.add_scalar('val/precision', precision, self.val_iter_number)
            self.summary.add_scalar('val/recall', recall, self.val_iter_number)
            self.summary.add_scalar('val/loss: ', loss_eval, self.val_iter_number)

            print("val epoch:  {}/{}, ith:  {}/{}, loss：  {:.4f}, mIOU:  {:.2%}, accuracy  {:.2%}，precision  {:.2%}，recall  {:.2%}"
                  .format(epoch_, self.config['epoch'], ith, len(self.val_data), loss_eval, mIOU, Acc, precision, recall))
        average_loss = np.nanmean(epoch_loss)
        average_IOU = np.nanmean(IOU_epoch, axis=0)
        average_mIOU = np.nanmean(mIOU_epoch)
        average_Acc = np.nanmean(Acc_epoch)
        average_precision = np.nanmean(precision_epoch)
        average_recall = np.nanmean(recall_epoch)

        self.summary.add_scalar('val/acc_epoch', average_Acc, epoch_)
        self.summary.add_scalar('val/loss_epoch', average_loss, epoch_)
        self.summary.add_scalar('val/mIOU_epoch', average_mIOU, epoch_)
        self.summary.add_scalar('val/precision_epoch', average_precision, epoch_)
        self.summary.add_scalar('val/recall_epoch', average_recall, epoch_)

        print("epoch:  {}/{}, average_loss： {:.4f}，average_mIOU:  {:.2%}, average_accuracy：  {:.2%}, average_precision：  {:.2%}, average_recall：  {:.2%}"
              .format(epoch_, self.config['epoch'], average_loss, average_mIOU, average_Acc, average_precision, average_recall))
        print("class name  ", self.class_name)
        print("IOU/class:  ", average_IOU)
        print('------------------------------------------------------------------')
        if average_mIOU > self.best_pred:
            self.best_pred = average_mIOU
            self.save(epoch_)

    def load(self):
        load_path = os.path.join(self.config["resume_path"], 'parameters.pth')
        if os.path.isfile(load_path):
            load_parameters = torch.load(load_path, map_location=lambda storage, loc: storage.cuda(self.device))
            self.optimizer.load_state_dict(load_parameters['optimizer'])
            self.lr_scheduler.load_state_dict(load_parameters['lr_scheduler'])
            self.model.load_state_dict(load_parameters['model'])
            self.best_pred = load_parameters['best_prediction']
            self.epoch = load_parameters['epoch']
            self.train_iter_number = load_parameters['train_iter_number']
            self.val_iter_number = load_parameters['val_iter_number']
            self.loss_value = load_parameters['loss_value']
            self.model = self.model.to(self.device)
            self.lr_scheduler.step(self.train_iter_number)

    def save(self, epoch_):
        os.mkdir(self.config["resume_path"]) if not os.path.exists(self.config["resume_path"]) else None
        torch.save({'best_prediction': self.best_pred,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': epoch_,
                    'train_iter_number': self.train_iter_number,
                    'val_iter_number': self.val_iter_number,
                    'loss_value': self.loss_value},
                   os.path.join(self.config["resume_path"], 'parameters.pth'))

    def data_preprocess(self, data_dict, data_type='train'):
        coords = data_dict[0]
        feats = data_dict[1]/255-0.5
        labels = data_dict[2]
        length = coords.shape[0]

        if length > self.point_number and data_type == 'train':
            inds = np.random.choice(np.arange(length), self.point_number, replace=False)
            coords, feats, labels = coords[inds], feats[inds], labels[inds]

        if data_type == 'train':
            # For some networks, making the network invariant to even, odd coords is important
            coords[:, :3] += (torch.rand(3) * 100).type_as(coords)

        points = ME.SparseTensor(feats, coords.int())

        if self.config["use_cuda"]:
            points, labels = points.to(self.device), labels.to(self.device)
        return points, labels.long()

    def reset_loss(self, label):
        data_ = label.cpu().numpy()
        number = data_.shape[0]
        bin = np.bincount(data_, minlength=self.config["class"])
        for ith, i in enumerate(bin):
            if i != 0:
                bin[ith] = number/i
        bin = bin/np.sum(bin)
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(bin).cuda().float(), ignore_index=self.config['ignore_label'])

