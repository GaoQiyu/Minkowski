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
        if self.config["use_cuda"]:
            self.model = self.model.cuda(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["lr"], momentum=self.config["momentum"],
                                         weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.loss = torch.nn.CrossEntropyLoss()

        self.train_data = S3DISDataset(self.config["data_path"], voxel_size=self.config["voxel_size"])
        self.val_data = S3DISDataset(self.config["data_path"], data_type='val', voxel_size=config["voxel_size"])
        # self.train_data_loader = DataLoader(train_data, self.config["batch_size"], shuffle=True)
        # self.val_data_loader = DataLoader(val_data, self.config["batch_size"])

        log_path = os.path.join(config["log_path"], str(time.time()))
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.summary = SummaryWriter(log_path)
        self.evaluator = Evaluator(self.config["class"])

        self.load()

    def train(self, epoch_):
        epoch_loss = 0
        self.model.train()
        for ith, samples in enumerate(self.train_data):
            # coords, feats, label = samples
            # point = ME.SparseTensor(feats, coords, batch=True)
            point, label = samples
            if self.config["use_cuda"]:
                point, label = point.to(self.device), label.to(self.device).long()

            output_sparse = self.model(point)

            pred = output_sparse.F
            loss = self.loss(pred, label)
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
            point, label = samples
            if self.config["use_cuda"]:
                point, label = point.to(self.device), label
            with torch.no_grad():
                output = self.model(point)
            pred = output.F.max(1)[1]
            IOU, mIOU = self.evaluator.mIOU(pred.cpu().double(), label)
            mIOU_epoch += mIOU
            self.summary.add_scalar('val/mIOU', mIOU, epoch_*len(self.val_data)+ith)
            print("val epoch:  {}/{}, ith:  {}/{}, mIOU:  {}%".format(epoch_, self.config['epoch'], ith, len(self.val_data), mIOU*100))
        average_mIOU = mIOU_epoch/len(self.val_data)
        self.summary.add_scalar('val/mIOU_epoch', average_mIOU, epoch_)
        print("epoch:    {}/{}, average_mIOU:    {}%".format(epoch_, self.config['epoch'], average_mIOU*100))
        print('------------------------------------------------------------------')
        if average_mIOU < self.best_pred:
            self.save(epoch_)

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
        if not os.path.exists(config["resume_path"]):
            os.mkdir(config["resume_path"])
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
        trainer.train(epoch)
        trainer.eval(epoch)
        trainer.summary.close()
