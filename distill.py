
import torch.nn as nn
import torch
import os
import numpy as np
from unet import unet_s, unet_m, unet_l
from data import SegData
import torch
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from algorithms.FitNet import loss_kd_fitnet
from loguru import logger

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
def make_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_root', type=str, default='data/train', help='architecture name')
    parser.add_argument('--val_root', type=str, default='data/val', help='architecture name')
    parser.add_argument('--n_epoch', type=int, default=200, help='architecture name')
    parser.add_argument('--batch_size', type=int, default=64, help='architecture name')
    parser.add_argument('--log_freq', type=int, default=100, help='architecture name')
    parser.add_argument('--val_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_dir', type=str, default='output', help='architecture name')
    parser.add_argument('--lr', type=float, default=1e-4, help='architecture name')
    parser.add_argument('--T', type=float, default=20, help='temperature for knowledge distillation')
    parser.add_argument('--alpha', type=float, default=0.9, help='weight for distillation loss')
    parser.add_argument('--pretrained', type=str, default='./output_ks_64_unetms/99.pth', help='path of the pretrained model')
    parser.add_argument('--distilled', type=str, default='distilled', help='path of the distilled model')
    parser.add_argument('--KD', type=str, default='FN', help='KD moethods')
    return parser.parse_args()

class Distiller:
    def __init__(self, args):
        self.args=args

        os.makedirs(args.save_dir, exist_ok=True)

        self.build_model()
        self.build_data()

    def build_model(self):
        self.teacher_model = unet_m(1, 3).to(device)
        self.student_model = unet_s(1, 3).to(device)
        if self.args.KD == 'KD':
            self.criterion = nn.KLDivLoss(reduction='batchmean')

        if self.args.KD == 'MSE':
            self.criterion = nn.MSELoss()

        if self.args.KD == 'MSE':
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=self.args.lr)

        if self.args.pretrained:
            state_dict = torch.load(self.args.pretrained, map_location=device)
            self.teacher_model.load_state_dict(state_dict)

    def build_data(self):
        dataset_train = SegData(self.args.train_root)
        self.data_loader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        dataset_val = SegData(self.args.val_root)
        self.data_loader_val = DataLoader(dataset=dataset_val, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    def evaluate(self):
        self.student_model.eval()

        acc_group = np.array([0, 0, 0])
        data_count = np.array([0, 0, 0])
        for data, label in self.data_loader_val:
            data = data.to(device)
            label = label.to(device)

            data_count += len(label)

            output = self.student_model(data)
            pred = torch.argmax(output, dim=1)

            for i in range(3):
                idx_cls = label == i
                data_count[i] += idx_cls.sum()
                acc_group[i] += (label[idx_cls] == pred[idx_cls]).sum().item()

        m_acc = acc_group / data_count
        logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)}')
        return m_acc

    def save_model(self, epoch):
        path = os.path.join(self.args.save_dir, f"model_{epoch + 1}.pth")
        torch.save(self.student_model.state_dict(), path)
        logger.info(f"Saved model checkpoint at epoch {epoch + 1}")

    def train(self):
        for epoch in range(self.args.n_epoch):
            self.student_model.train()
            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device) #[B, 1, N]
                label = label.to(device)
                self.teacher_model.eval()
                teacher_output = self.teacher_model(data)
                student_output = self.student_model(data)
                self.teacher_model.eval()
                loss_ce = nn.CrossEntropyLoss()(student_output, label)
                if self.args.KD == 'KD':
                    loss_kd = self.criterion(F.log_softmax(student_output/self.args.T, dim=1), F.softmax(teacher_output/self.args.T, dim=1))
                    loss = self.args.alpha * loss_kd + (1 - self.args.alpha) * loss_ce
                if self.args.KD == 'MSE':
                    loss_kd = self.criterion(student_output, teacher_output)
                    loss = loss_ce * (1.0 - self.args.alpha) + loss_kd * self.args.alpha
                if self.args.KD == 'FN':
                    loss = loss_kd_fitnet(teacher_output,student_output,args.T,alpha=args.alpha)



                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % self.args.log_freq == 0:
                    logger.info(
                        f"Epoch [{epoch + 1}/{self.args.n_epoch}] Step [{step}/{len(self.data_loader_train)}] Loss: {loss.item()}")

            if (epoch + 1) % self.args.val_freq == 0:
                self.evaluate()

            if (epoch + 1) % self.args.save_freq == 0:
                self.save_model(epoch)



if __name__ == '__main__':
    args = make_args()
    distiller = Distiller(args)
    distiller.train()
