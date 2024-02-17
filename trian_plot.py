import torch.nn as nn
import torch.nn.functional as F
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
    parser.add_argument('--n_epoch', type=int, default=500, help='architecture name')
    parser.add_argument('--batch_size', type=int, default=256, help='architecture name')
    parser.add_argument('--log_freq', type=int, default=10, help='architecture name')
    parser.add_argument('--val_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_freq', type=int, default=20, help='architecture name')
    parser.add_argument('--save_dir', type=str, default='output_s', help='architecture name')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    return parser.parse_args()


class Distiller(nn.Module):
    def __init__(self, teacher, student):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x):
        teacher_output = self.teacher(x)
        student_output = self.student(x)
        loss = self.criterion(F.log_softmax(student_output / 3.0, dim=1), F.softmax(teacher_output / 3.0, dim=1))
        return loss


def plot_loss_and_acc(train_loss, val_acc, val_loss):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    # plot training loss
    axes[0].plot(train_loss)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    # plot validation acc
    axes[1].plot(val_acc)
    axes[1].set_title('Validation Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Acc')

    # plot validation loss
    axes[2].plot(val_loss)
    axes[2].set_title('Validation Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')

    # plot validation accuracy for each class
    # for i in range(3):
    #     axes[i + 2].plot(val_acc[:, i])
    #     axes[i + 2].set_title(f'Validation Accuracy for Class {i}')
    #     axes[i + 2].set_xlabel('Epoch')
    #     axes[i + 2].set_ylabel('Accuracy')

    # adjust layout and save figure
    fig.tight_layout()
    fig.savefig('output_s/loss_and_acc_kd.png')


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
        # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([3, 2, 3], dtype=torch.float32).to(device))
        self.optimizer_teacher = torch.optim.AdamW(self.model_teacher.parameters(), lr=self.args.lr)
        self.optimizer_student = torch.optim.AdamW(self.model_student.parameters(), lr=self.args.lr)

    def build_data(self):
        dataset_train = SegData(self.args.train_root)
        self.data_loader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        dataset_val = SegData(self.args.val_root)
        self.data_loader_val = DataLoader(dataset=dataset_val, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    def train(self):

        train_loss = []
        val_acc = []
        val_loss= []
        for epoch in range(self.args.n_epoch):
            self.model_teacher.train()
            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device) #[B, 1, N]
                label = label.to(device)

                output = self.model_teacher(data)

                loss = self.criterion(output, label)

                train_loss.append(loss.item())

                self.optimizer_teacher.zero_grad()
                loss.backward()
                self.optimizer_teacher.step()

                if (step+1)%self.args.log_freq==0:
                    logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')

            if (epoch+1)%self.args.val_freq==0:
                acc, loss = self.val()
                val_acc.append(np.sum(acc))
                val_loss.append(loss.item())
                if np.sum(acc) >= np.sum(self.best_acc):
                    self.best_acc = acc
                    torch.save(self.model_teacher.state_dict(), os.path.join(self.args.save_dir, 'best_model.pth'))

            if (epoch+1)%self.args.save_freq==0:
                torch.save(self.model_teacher.state_dict(), os.path.join(self.args.save_dir, f'{epoch}.pth'))

            # distillation
            self.model_teacher.eval()
            self.model_student.train()
            distiller = Distiller(self.model_teacher, self.model_student)
            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device)
                label = label.to(device)

                loss = distiller(data)

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

            logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, distillation loss: {loss.item()}')

        # plot loss and accuracy
        plot_loss_and_acc(train_loss, val_acc , val_loss)


    @torch.no_grad()
    def val(self):

        self.model_teacher.eval()

        acc_group=np.array([0,0,0])
        data_count=np.array([0,0,0])
        for data, label in self.data_loader_val:
            data = data.to(device)
            label = label.to(device)

            data_count+=len(label)

            output = self.model_teacher(data)

            loss = self.criterion(output,label)

            pred=torch.argmax(output, dim=1)

            for i in range(3):
                idx_cls = label==i
                data_count[i]+=idx_cls.sum()
                acc_group[i]+=(label[idx_cls]==pred[idx_cls]).sum().item()

        m_acc = acc_group/data_count

        logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)},val_loss: {loss.item()}')
        # logger.info(f'val_loss: {loss.item()}')
        return m_acc, loss


if __name__ == '__main__':
    args=make_args()
    trainer=Trainer(args)
    trainer.train()
