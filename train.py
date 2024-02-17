import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from unet import unet_s, unet_m, unet_l
from data import SegData
import torch
import argparse
from torch.utils.data import DataLoader

from loguru import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_root', type=str, default='data/train', help='architecture name')
    parser.add_argument('--val_root', type=str, default='data/val', help='architecture name')
    parser.add_argument('--n_epoch', type=int, default=2000, help='architecture name')
    parser.add_argument('--batch_size', type=int, default=512, help='architecture name')
    parser.add_argument('--log_freq', type=int, default=100, help='architecture name')
    parser.add_argument('--val_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_freq', type=int, default=20, help='architecture name')
    parser.add_argument('--save_dir', type=str, default='output_without_distill_kd', help='architecture name')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    return parser.parse_args()


class Trainer:
    def __init__(self, args):
        self.args = args
        self.best_acc = 0.0

        os.makedirs(args.save_dir, exist_ok=True)

        self.build_model()
        self.build_data()

    def build_model(self):
        self.model_teacher = unet_m(1, 3).to(device)
        self.model_student = unet_s(1, 3).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model_student.parameters(), lr=self.args.lr)

    def build_data(self):
        dataset_train = SegData(self.args.train_root)
        self.data_loader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        dataset_val = SegData(self.args.val_root)
        self.data_loader_val = DataLoader(dataset=dataset_val, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=8)


    def train(self):
        for epoch in range(self.args.n_epoch):
            self.model_student.train()
            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device) #[B, 1, N]
                label = label.to(device)

                output = self.model_student(data)

                loss = self.criterion(output, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (step+1)%self.args.log_freq==0:
                    logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')

            if (epoch+1)%self.args.val_freq==0:
                acc = self.val()
                if np.sum(acc) > np.sum(self.best_acc):
                    self.best_acc = acc
                    torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir, 'best_model.pth'))

            if (epoch+1)%self.args.save_freq==0:
                torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir, f'{epoch}.pth'))



    @torch.no_grad()
    def val(self):
        self.model_student.eval()

        acc_group=np.array([0,0,0])
        data_count=np.array([0,0,0])
        for data, label in self.data_loader_val:
            data = data.to(device)
            label = label.to(device)

            data_count+=len(label)

            output = self.model_student(data)
            pred=torch.argmax(output, dim=1)

            for i in range(3):
                idx_cls = label==i
                data_count[i]+=idx_cls.sum()
                acc_group[i]+=(label[idx_cls]==pred[idx_cls]).sum().item()

        m_acc = acc_group/data_count
        logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)}')
        return m_acc




if __name__ == '__main__':
    args=make_args()
    trainer=Trainer(args)
    trainer.train()
