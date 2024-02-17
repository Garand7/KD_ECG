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
#%%
def make_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_root', type=str, default='data/train', help='architecture name')
    parser.add_argument('--val_root', type=str, default='data/val', help='architecture name')
    parser.add_argument('--n_epoch', type=int, default=200, help='architecture name')
    parser.add_argument('--batch_size', type=int, default=12800, help='architecture name')
    parser.add_argument('--log_freq', type=int, default=10, help='architecture name')
    parser.add_argument('--val_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_freq', type=int, default=20, help='architecture name')
    parser.add_argument('--save_dir', type=str, default='output_m', help='architecture name')
    parser.add_argument('--save_dir_kd', type=str, default='output_s', help='architecture name')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_kd', type=float, default=5e-4, help='learning rate')
    return parser.parse_args()

class Attention(nn.Module):
    """
    空间注意力, 通过学习通道之间的权重来增强网络的表达能力
    """
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction = reduction

        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1)

    def forward(self, x):
        _,_,H,W = x.size() # 输入数据的h和w(这里假设通道数为c)
        x_mean = x.mean(dim=1, keepdim=True) # N * 1 * H * W
        x_mean = self.conv1(x_mean) # N * 1/4*c * H * W
        x_mean = F.relu(x_mean)
        x_mean = self.conv2(x_mean) # N * c*out_channels * H * W
        x_mean = F.sigmoid(x_mean) # Sigmoid激活函数, 用于压缩权重的值
        x_mean = x_mean.view(-1, self.out_channels, H, W)
        return x * x_mean.expand_as(x)

class KD_loss_with_attention(nn.Module):
    def __init__(self, alpha, temperature, reduction):
        super(KD_loss_with_attention, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.reduction = reduction
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, student_outputs, teacher_outputs):
        """
        计算loss并返回

        :param inputs:
        :param student_outputs:
        :param teacher_outputs:
        :return: loss
        """
        def feature_flatten(inputs, reduction):
            bs, ch, h, w = inputs.size()
            inputs = inputs.view(bs, ch, -1)
            inputs = torch.mean(inputs, dim=2, keepdim=True)  # sum over H * W
            inputs = inputs.pow(self.temperature)
            if reduction == 'softmax':
                inputs = F.softmax(inputs, dim=1)  # normalize along C
            elif reduction == 'sum':
                inputs = inputs.sum(dim=1).mean()  # sum over C, mean over B
            elif reduction == 'none':
                inputs = inputs.view(bs, -1)
            return inputs

        loss = nn.KLDivLoss()(F.log_softmax(student_outputs / self.temperature, dim=1),
                              F.softmax(teacher_outputs / self.temperature, dim=1)) * self.temperature * self.temperature * self.alpha

        student_attention = Attention(student_outputs.size()[1], student_outputs.size()[1] // 16)
        teacher_attention = Attention(teacher_outputs.size()[1], teacher_outputs.size()[1] // 16)

        student_feature = feature_flatten(student_outputs, reduction=self.reduction).detach()
        teacher_feature = feature_flatten(teacher_outputs, reduction=self.reduction)
        weight = F.softplus(teacher_feature)
        student_attention_map = student_attention(student_outputs * weight)
        teacher_attention_map = teacher_attention(teacher_outputs)
        loss_attention = self.mse_loss(student_attention_map, teacher_attention_map)
        loss += loss_attention

        return loss

def plot_loss_and_acc(train_loss, val_acc, val_loss,dir):
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
    fig.savefig(dir+'/loss_and_acc.png')



class Trainer:
    def __init__(self, args):
        self.args = args
        self.best_acc = 0.0

        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir_kd, exist_ok=True)


        self.build_model()
        self.build_data()

    def build_model(self):
        self.model_teacher = unet_m(1, 3).to(device)
        self.model_student = unet_s(1, 3).to(device)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([5, 7, 8], dtype=torch.float32).to(device))
        self.optimizer_teacher = torch.optim.AdamW(self.model_teacher.parameters(), lr=self.args.lr)
        self.optimizer_student = torch.optim.AdamW(self.model_student.parameters(), lr=self.args.lr_kd)

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
        # plot loss and accuracy
        plot_loss_and_acc(train_loss, val_acc , val_loss,dir=args.save_dir)


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



            # val_loss.append(loss.item())
        m_acc = acc_group/data_count

        logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)},val_loss: {loss.item()}')
        # logger.info(f'val_loss: {loss.item()}')
        return m_acc, loss

    def val_student(self):

        self.model_student.eval()

        acc_group = np.array([0, 0, 0])
        data_count = np.array([0, 0, 0])
        for data, label in self.data_loader_val:
            data = data.to(device)
            label = label.to(device)

            data_count += len(label)

            output = self.model_student(data)

            loss = self.criterion(output, label)

            pred = torch.argmax(output, dim=1)

            for i in range(3):
                idx_cls = label == i
                data_count[i] += idx_cls.sum()
                acc_group[i] += (label[idx_cls] == pred[idx_cls]).sum().item()

            # val_loss.append(loss.item())
        m_acc = acc_group / data_count

        logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)},val_loss: {loss.item()}')
        # logger.info(f'val_loss: {loss.item()}')
        return m_acc, loss

    def distillation_kd(self, T , alpha):

        train_loss_kd = []
        val_acc_kd = []
        val_loss_kd = []

        for epoch in range(1000):

            self.model_teacher.eval()
            self.model_student.train()

            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device) #[B, 1, N]
                label = label.to(device)

                output_teacher = self.model_teacher(data)
                output_student = self.model_student(data)

                loss_ce = self.criterion(output_student,label)

                loss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_student/T, dim=1), F.softmax(output_teacher/T, dim=1))*T*T

                loss = alpha*loss_kd + (1-alpha)*loss_ce
                train_loss_kd.append(loss.item())

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                if (step+1)%self.args.log_freq==0:
                    logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')

            if (epoch + 1) % self.args.val_freq == 0:
                acc, loss = self.val_student()
                val_acc_kd.append(np.sum(acc))
                val_loss_kd.append(loss.item())
                if np.sum(acc) >= np.sum(self.best_acc):
                    self.best_acc = acc
                    torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir_kd, 'best_model.pth'))

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir_kd, f'{epoch}.pth'))
                # plot loss and accuracy
            plot_loss_and_acc(train_loss_kd, val_acc_kd, val_loss_kd,dir=args.save_dir_kd)

if __name__ == '__main__':
    args=make_args()
    trainer=Trainer(args)
    trainer.train()
    trainer.distillation_kd(T=3, alpha=0.2)