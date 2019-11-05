import os
import json
import torch
import math
import time
import numpy as np
import MinkowskiEngine as ME
import model.pointnet as PointNet
import model.res16unet as ResUNet
from model.lr_scheduler import PolyLR
from data import dataloader
from model.evaluator import Evaluator
from tensorboardX import SummaryWriter


class Trainer(object):
    def __init__(self, config_):
        self.config = config_
        self.best_pred = -math.inf
        self.train_iter_number = 0
        self.val_iter_number = 0
        self.epoch = 0
        self.device = torch.device(0)
        self.loss_value = torch.tensor(0.0, requires_grad=True).to(self.device)
        self.point_number = self.config["point_num"]
        self.batch_size = self.config["batch_size"]
        # self.model = PointNet.PointNet(self.config["class"])
        self.model = ResUNet.Res16UNet34C(3, self.config["class"])
        if self.config["fine_tune"]:
            model_dict = torch.load(os.path.join(config["resume_path"], 'weights_14.pth'), map_location=lambda storage, loc: storage.cuda(self.device))
            self.model.load_state_dict(model_dict)
            self.optimizer = torch.optim.SGD([
                {'params': self.model.convtr7p2s2.parameters(), 'lr': self.config["lr"] / 1e2},
                {'params': self.model.bntr7.parameters(), 'lr': self.config["lr"] / 1e2},
                {'params': self.model.block8.parameters(), 'lr': self.config["lr"] / 1e1},
                {'params': self.model.final.parameters(), 'lr': self.config["lr"]}],
                lr=self.config["lr"] / 1e4, momentum=self.config["momentum"], weight_decay=1e-4)
        if self.config["use_cuda"]:
            self.model = self.model.to(self.device)

        self.loss = torch.nn.CrossEntropyLoss(ignore_index=self.config['ignore_label'])

        self.train_data = dataloader(1, self.config["data_path"], voxel_size=self.config["voxel_size"], transform=False, shuffle=True)
        self.val_data = dataloader(1, self.config["data_path"], data_type='val', voxel_size=config["voxel_size"])

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['lr'],
        #                                  momentum=self.config['momentum'], weight_decay=self.config['weight_decay'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['lr'],
                                          weight_decay=self.config['weight_decay'])

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'], gamma=0.1, last_epoch=-1)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3)
        self.lr_scheduler = PolyLR(self.optimizer, max_iter=self.config['epoch']*len(self.train_data), power=self.config['poly_power'], last_step=-1)

        log_path = os.path.join(config["log_path"], str(time.time()))
        os.mkdir(log_path) if not os.path.exists(log_path) else None
        self.summary = SummaryWriter(log_path)
        self.evaluator = Evaluator(self.config["class"])

        self.load()

    def train(self, epoch_):
        epoch_loss = 0
        self.model.train()
        for ith, data_dict in enumerate(self.train_data):
            point, labels = self.data_preprocess(data_dict)
            output_sparse = self.model(point)
            pred = output_sparse.F
            self.loss_value = self.loss(pred, labels) + self.loss_value
            epoch_loss += self.loss(pred, labels).item() / self.config["accumulate_gradient"]
            self.train_iter_number += 1
            if self.train_iter_number % self.batch_size == 0:
                self.loss_value /= (self.config["accumulate_gradient"]*self.batch_size)
                self.loss_value.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.summary.add_scalar('train/loss: ', self.loss_value, self.train_iter_number // self.batch_size)
                print("train epoch:  {}/{}, ith:  {}/{}, loss:  {}".format(epoch_, self.config['epoch'], ith, len(self.train_data), self.loss_value.item()))
                self.loss_value = 0
        average_loss = epoch_loss/len(self.train_data)
        # StepLR
        self.lr_scheduler.step(epoch_)

        self.summary.add_scalar('train/loss_epoch: ', average_loss, epoch_)
        print("epoch:    {}/{}, average_loss:    {}".format(epoch_, self.config['epoch'], average_loss))
        print('------------------------------------------------------------------')

    def eval(self, epoch_):
        self.model.eval()
        mIOU_epoch = 0
        accuracy_epoch = 0
        precision_epoch = 0
        recall_epoch = 0
        epoch_loss = 0
        for ith, data_dict in enumerate(self.val_data):
            point, labels = self.data_preprocess(data_dict)

            with torch.no_grad():
                output = self.model(point)
            pred = output.F
            loss_eval = self.loss(pred, labels) / self.config["accumulate_gradient"]
            epoch_loss += loss_eval.item()
            mIOU, accuracy, precision, recall = self.evaluator.generate(pred.max(1)[1].cpu(), labels.cpu())
            mIOU_epoch += mIOU
            accuracy_epoch += accuracy
            precision_epoch += precision
            recall_epoch += recall

            self.val_iter_number += 1
            self.summary.add_scalar('val/mIOU', mIOU, self.val_iter_number)
            self.summary.add_scalar('val/accuracy', accuracy, self.val_iter_number)
            self.summary.add_scalar('val/precision', precision, self.val_iter_number)
            self.summary.add_scalar('val/recall', recall, self.val_iter_number)
            self.summary.add_scalar('val/loss: ', loss_eval, self.val_iter_number)

            print("val epoch:  {}/{}, ith:  {}/{}, loss：  {}, mIOU:  {}%, accuracy  {}%，precision  {}%，recall  {}%"
                  .format(epoch_, self.config['epoch'], ith, len(self.val_data), loss_eval, mIOU*100, accuracy*100, precision*100, recall*100))
        average_loss = epoch_loss / len(self.val_data)
        average_mIOU = mIOU_epoch/len(self.val_data)
        average_accuracy = accuracy_epoch / len(self.val_data)
        average_precision = precision_epoch / len(self.val_data)
        average_recall = recall_epoch / len(self.val_data)

        # ReduceLROnPlateau
        self.lr_scheduler.step(average_loss)

        self.summary.add_scalar('val/loss_epoch', average_loss, epoch_)
        self.summary.add_scalar('val/mIOU_epoch', average_mIOU, epoch_)
        self.summary.add_scalar('val/accuracy_epoch', average_accuracy, epoch_)
        self.summary.add_scalar('val/precision_epoch', average_precision, epoch_)
        self.summary.add_scalar('val/recall_epoch', average_recall, epoch_)

        print("epoch:  {}/{}, average_loss： {}，average_mIOU:  {}%, average_accuracy：  {}%, average_precision：  {}%, average_recall：  {}%"
              .format(epoch_, self.config['epoch'], average_loss, average_mIOU * 100, average_accuracy * 100, average_precision * 100, average_recall * 100))
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

    def save(self, epoch_):
        os.mkdir(config["resume_path"]) if not os.path.exists(config["resume_path"]) else None
        torch.save({'best_prediction': self.best_pred,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': epoch_,
                    'train_iter_number': self.train_iter_number,
                    'val_iter_number': self.val_iter_number,
                    'loss_value': self.loss_value},
                   os.path.join(self.config["resume_path"], 'parameters.pth'))

    def data_preprocess(self, data_dict):
        coords = data_dict['coords']
        feats = data_dict['feats']
        labels = data_dict['labels']

        coords[:, :3] = np.floor(coords[:, :3] / self.config['voxel_size'])
        inds = ME.utils.sparse_quantize(coords[:, :3].numpy(), return_index=True)

        if len(inds) > self.point_number:
            inds = np.random.choice(inds, self.point_number, replace=False)
        coordinates, features, labels = coords[inds], feats[inds], labels[inds]

        # # For some networks, making the network invariant to even, odd coords is important
        # coordinates[:, :3] += (torch.rand(3) * 100).type_as(coordinates)

        points = ME.SparseTensor(features, coordinates.int())

        if self.config["use_cuda"]:
            points, labels = points.to(self.device), labels.to(self.device)
        return points, labels


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath("./"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()

    trainer = Trainer(config)
    time_now = time.time()
    for epoch in range(trainer.epoch, config["epoch"]):
        if config["multiple_fold"]:
            trainer.train_data = dataloader(1, config["data_path"], fold=epoch%3, voxel_size=config["voxel_size"], transform=True, shuffle=True)
            trainer.val_data = dataloader(1, config["data_path"], fold=epoch%3, data_type='val', voxel_size=config["voxel_size"])
        trainer.train(epoch)
        trainer.eval(epoch)
        print('one epoch time:   {} s'.format(time.time() - time_now))
        trainer.summary.close()
        time_now = time.time()

