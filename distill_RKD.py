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
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
class RKDLoss(nn.Module):

    def __init__(self, alpha, T):
        super(RKDLoss, self).__init__()
        self.alpha = alpha
        self.T = T

    def forward(self, student_features, teacher_features):
        student_gram = gram_matrix(student_features)
        teacher_gram = gram_matrix(teacher_features)
        loss_value = ((self.alpha/self.T)**2) * torch.mean((teacher_gram - student_gram)**2)
        return loss_value

def gram_matrix(x):
    """
    Compute the Gram matrix of input x.
    """
    x = x.view(x.size(0), -1)
    gram = torch.mm(x, x.t())
    return gram
#%%
def rkd_loss(student_features, teacher_features, alpha, beta):
    def pairwise_distances(x, y=None):
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
        return torch.clamp(dist, 0.0, np.inf)

    def loss_func(X, Y):
        dist = pairwise_distances(X, Y)
        exp = torch.exp(-dist/beta)
        A_row = torch.sum(exp, dim=1)
        A_col = torch.sum(exp, dim=0)
        loss = torch.mean(torch.abs(A_row - A_col))
        return loss

    loss = 0
    for (student, teacher) in zip(student_features, teacher_features):
        loss += alpha * loss_func(student, student)
        loss += (1 - alpha) * loss_func(student, teacher)
    return loss
#%%
def make_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_root', type=str, default='data/train', help='architecture name')
    parser.add_argument('--val_root', type=str, default='data/val', help='architecture name')
    parser.add_argument('--n_epoch', type=int, default=300, help='architecture name')
    parser.add_argument('--n_epoch_s', type=int, default=4000, help='architecture name')
    parser.add_argument('--batch_size', type=int, default=12800, help='architecture name')
    parser.add_argument('--log_freq', type=int, default=10, help='architecture name')
    parser.add_argument('--val_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_freq', type=int, default=20, help='architecture name')
    parser.add_argument('--save_dir', type=str, default='output_m_rkd', help='architecture name')
    parser.add_argument('--save_dir_rkd', type=str, default='output_s_rkd_75k', help='architecture name') #output_s_rkd_175
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_rkd', type=float, default=2e-4, help='learning rate')
    return parser.parse_args()

def plot_loss_and_acc(train_loss, train_acc, val_loss, val_acc,dir):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

    # plot training loss
    axes[0].plot(train_loss)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    # plot train acc
    axes[1].plot(train_acc)
    axes[1].set_title('Train Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Acc')

    # plot validation acc
    axes[3].plot(val_acc)
    axes[3].set_title('Validation Acc')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Acc')

    # plot validation loss
    axes[2].plot(val_loss)
    axes[2].set_title('Validation Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')

    # adjust layout and save figure
    fig.tight_layout()
    fig.savefig(dir+'/loss_and_acc.png')


def save_metrics_to_excel(train_loss, train_acc, val_loss, val_acc, save_dir):
    # 创建一个pandas DataFrame来存储数据
    df = pd.DataFrame({
        'Epoch': list(range(1, len(train_loss) + 1)),
        'Train Loss': train_loss,
        'Train Acc': train_acc,
        'Validation Loss': val_loss,
        'Validation Acc': val_acc
    })

    # 设置Epoch列作为索引
    df.set_index('Epoch', inplace=True)

    # 保存DataFrame到Excel文件
    excel_path = os.path.join(save_dir, 'training_metrics.xlsx')
    df.to_excel(excel_path, engine='openpyxl')

    logger.info(f'Metrics saved to {excel_path}')

class Trainer:
    def __init__(self, args):
        self.args = args
        self.best_acc = 0.0

        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir_rkd, exist_ok=True)


        self.build_model()
        self.build_data()

    def build_model(self):
        self.model_teacher = unet_m(1, 3).to(device)
        self.model_student = unet_s(1, 3).to(device)
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([5, 7, 9], dtype=torch.float32).to(device))
        self.optimizer_teacher = torch.optim.AdamW(self.model_teacher.parameters(), lr=self.args.lr)
        self.optimizer_student = torch.optim.AdamW(self.model_student.parameters(), lr=self.args.lr_rkd)

    def build_data(self):
        dataset_train = SegData(self.args.train_root)
        self.data_loader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=8)

        dataset_val = SegData(self.args.val_root)
        self.data_loader_val = DataLoader(dataset=dataset_val, batch_size=self.args.batch_size, shuffle=False, pin_memory=True, num_workers=8)


    def train(self):
        train_acc = []
        train_loss = []
        val_acc = []
        val_loss= []
        acc_group = np.array([0, 0, 0])
        data_count = np.array([0, 0, 0])
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
                # acc
                data_count += len(label)
                pred = torch.argmax(output, dim=1)

                for i in range(3):
                    idx_cls = label == i
                    data_count[i] += idx_cls.sum()
                    acc_group[i] += (label[idx_cls] == pred[idx_cls]).sum().item()

                if (step+1)%self.args.log_freq==0:
                    logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')

            macc = acc_group / data_count
            train_acc.append(macc)

            if (epoch+1)%self.args.val_freq==0:
                acc, loss = self.val()
                val_acc.append(acc)
                val_loss.append(loss.item())
                if np.sum(acc) >= np.sum(self.best_acc):
                    self.best_acc = acc
                    torch.save(self.model_teacher.state_dict(), os.path.join(self.args.save_dir, 'best_model.pth'))

            if (epoch+1)%self.args.save_freq==0:
                torch.save(self.model_teacher.state_dict(), os.path.join(self.args.save_dir, f'{epoch}.pth'))
        # plot loss and accuracy
        plot_loss_and_acc(train_loss, train_acc , val_loss , val_acc,dir=args.save_dir)

        save_metrics_to_excel(train_loss, train_acc, val_loss, val_acc, self.args.save_dir)


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

    def distillation_rkd(self, T, alpha, beta):
        self.best_acc = 0
        train_acc_rkd =[]
        train_loss_rkd = []
        val_acc_rkd = []
        val_loss_rkd = []
        # criterion = RKDLoss(alpha,T)
        for epoch in range(args.n_epoch_s):

            self.model_teacher.eval()
            self.model_student.train()
            acc_group = np.array([0, 0, 0])
            data_count = np.array([0, 0, 0])
            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device) #[B, 1, N]
                label = label.to(device)

                output_teacher = self.model_teacher(data)
                output_student = self.model_student(data)

                loss_ce = self.criterion(output_student,label)

                loss_kd = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_student/T, dim=1), F.softmax(output_teacher/T, dim=1))*T*T
                # loss_rkd = criterion(output_student,output_teacher)
                loss_rkd = rkd_loss(student_features=[F.normalize(output_student[i], dim=1) for i in range(len(output_student))],teacher_features=[F.normalize(output_teacher[i], dim=1) for i in range(len(output_teacher))],
                                   alpha=alpha, beta=beta)
                loss = alpha*loss_rkd + (1-alpha)*(loss_ce+loss_kd)
                train_loss_rkd.append(loss.item())

                self.optimizer_student.zero_grad()
                loss.backward()
                self.optimizer_student.step()

                # acc
                data_count += len(label)
                pred = torch.argmax(output_student, dim=1)

                for i in range(3):
                    idx_cls = label == i
                    data_count[i] += idx_cls.sum()
                    acc_group[i] += (label[idx_cls] == pred[idx_cls]).sum().item()

                if (step+1)%self.args.log_freq==0:
                    logger.info(f'[{self.args.n_epoch_s}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            logger.info(f'[{self.args.n_epoch_s}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            macc = acc_group / data_count
            train_acc_rkd.append(macc)

            if (epoch + 1) % self.args.val_freq == 0:
                acc, loss = self.val_student()
                val_acc_rkd.append(acc)




                val_loss_rkd.append(loss.item())
                if np.sum(acc) >= np.sum(self.best_acc):
                    self.best_acc = acc
                    torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir_rkd, 'best_model.pth'))

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir_rkd, f'{epoch}.pth'))
                # plot loss and accuracy
            plot_loss_and_acc(train_loss_rkd, train_acc_rkd,val_loss_rkd, val_acc_rkd,dir=args.save_dir_rkd)

            save_metrics_to_excel(train_loss_rkd, train_acc_rkd,val_loss_rkd, val_acc_rkd, self.args.save_dir_rkd)

if __name__ == '__main__':
    args=make_args()
    trainer=Trainer(args)
    trainer.train()
    trainer.distillation_rkd(T=5, alpha=0.6,beta=180)