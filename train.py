import os
import json
import torch
import math
import time
from tensorboardX import SummaryWriter
import model.minkunet as Minkowski
from data.S3DIS import S3DISDataset
from model.evaluator import Evaluator
import MinkowskiEngine as ME


class Trainer(object):
    def __init__(self, config_):
        self.config = config_
        self.best_pred = math.inf
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = self.config["batch_size"]
        self.model = Minkowski.MinkUNet34C(3, self.config["class"])
        if self.config["fine_tune"]:
            model_dict = torch.load(os.path.join(config["resume_path"], 'weights_14.pth'))
            self.model.load_state_dict(model_dict)
        self.model = self.model.cuda(self.device) if self.config["use_cuda"] else None

        self.optimizer = torch.optim.SGD([
            {'params': self.model.convtr7p2s2.parameters(), 'lr': self.config["lr"] / 1e2},
            {'params': self.model.bntr7.parameters(), 'lr': self.config["lr"] / 1e2},
            {'params': self.model.block8.parameters(), 'lr': self.config["lr"] / 1e1},
            {'params': self.model.final.parameters(), 'lr': self.config["lr"]}],
            lr=self.config["lr"] / 1e4, momentum=self.config["momentum"], weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.loss = torch.nn.CrossEntropyLoss()

        self.train_data = S3DISDataset(self.config["data_path"], voxel_size=self.config["voxel_size"])
        self.val_data = S3DISDataset(self.config["data_path"], data_type='val', voxel_size=config["voxel_size"])
        # self.train_data_loader = DataLoader(train_data, self.config["batch_size"], shuffle=True)
        # self.val_data_loader = DataLoader(val_data, self.config["batch_size"])

        log_path = os.path.join(config["log_path"], str(time.time()))
        os.mkdir(log_path) if not os.path.exists(log_path) else None
        self.summary = SummaryWriter(log_path)
        self.evaluator = Evaluator(self.config["class"])

        self.load()

    def train(self, epoch_):
        epoch_loss = 0
        self.model.train()
        for ith, samples in enumerate(self.train_data):
            # coords, feats, label = samples
            # point = ME.SparseTensor(feats, coords, batch=True)
            point = samples[0].to(self.device) if self.config["use_cuda"] else samples[0].cpu()
            label = samples[1].long()

            output_sparse = self.model(point)
            pred = output_sparse.F
            loss = self.loss(pred.cpu(), label)
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step(epoch_)
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            self.summary.add_scalar('train/loss: ', loss.item(), epoch_ * len(self.train_data) + ith)
            print("train epoch:  {}/{}, ith:  {}/{}, loss:  {}".format(epoch_, self.config['epoch'], ith, len(self.train_data), loss.item()))
        average_loss = epoch_loss/len(self.train_data)
        self.summary.add_scalar('train/loss_epoch: ', average_loss, epoch_)
        print("epoch:    {}/{}, average_loss:    {}".format(epoch_, self.config['epoch'], average_loss))
        print('------------------------------------------------------------------')

    def eval(self, epoch_):
        self.model.eval()
        mIOU_epoch = 0
        for ith, samples in enumerate(self.val_data):
            point = samples[0].to(self.device) if self.config["use_cuda"] else samples[0].cpu()
            label = samples[1].long()
            with torch.no_grad():
                output = self.model(point)
            pred = output.F.max(1)[1]
            IOU, mIOU = self.evaluator.mIOU(pred.cpu(), label)
            mIOU_epoch += mIOU
            self.summary.add_scalar('val/mIOU', mIOU, epoch_*len(self.val_data)+ith)
            print("val epoch:  {}/{}, ith:  {}/{}, mIOU:  {}%".format(epoch_, self.config['epoch'], ith, len(self.val_data), mIOU*100))
        average_mIOU = mIOU_epoch/len(self.val_data)
        self.summary.add_scalar('val/mIOU_epoch', average_mIOU, epoch_)
        print("epoch:    {}/{}, average_mIOU:    {}%".format(epoch_, self.config['epoch'], average_mIOU*100))
        print('------------------------------------------------------------------')
        self.save(epoch_) if average_mIOU < self.best_pred else None

    def load(self):
        load_path = os.path.join(self.config["resume_path"], 'parameters.pth')
        if os.path.isfile(load_path):
            load_parameters = torch.load(load_path)
            self.optimizer.load_state_dict(load_parameters['optimizer'])
            self.lr_scheduler.load_state_dict(load_parameters['lr_scheduler'])
            self.model.load_state_dict(load_parameters['model'])
            self.best_pred = load_parameters['best_prediction']
            self.model = self.model.cuda()

    def save(self, epoch_):
        os.mkdir(config["resume_path"]) if not os.path.exists(config["resume_path"]) else None
        torch.save({'best_prediction': self.best_pred,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'epoch': epoch_},
                   os.path.join(self.config["resume_path"], 'parameters.pth'))


if __name__ == '__main__':
    config_path = os.path.join(os.path.abspath("./"), "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config_file.close()

    trainer = Trainer(config)
    for epoch in range(config["epoch"]):
        if config["multiple_fold"] and (epoch == math.floor(config["epoch"]/3) or epoch == math.floor(config["epoch"]*2/3)):
            for fold_inds in range(1, 3):
                trainer.train_data = S3DISDataset(config["data_path"], fold=fold_inds,  voxel_size=config["voxel_size"])
                trainer.val_data = S3DISDataset(config["data_path"], fold=fold_inds, data_type='val', voxel_size=config["voxel_size"])
        # trainer.train(epoch)
        trainer.eval(epoch)
        trainer.summary.close()
