import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from unet import  unet_m, unet_l,unet_s
# from unet_s_dfscrd import unet_s
from data import SegData
import torch
import argparse
from torch.utils.data import DataLoader
from loguru import logger


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#%%
def CRD(t_feature, s_feature, temperature):
    T = temperature
    # Use Euclidean distance as the distance metric
    t_feature = t_feature.view(t_feature.size(0), -1)
    s_feature = s_feature.view(s_feature.size(0), -1)

    N, C = t_feature.size()[0], t_feature.size()[1]

    # calculate teacher similarity matrix
    t_similarity_matrix = torch.cdist(t_feature, t_feature, p=2)
    t_similarity_matrix = torch.exp(-t_similarity_matrix / T)
    t_similarity_matrix = torch.softmax(t_similarity_matrix, dim=-1)
    # dimension reduction on teacher similarity matrix
    c_t = torch.mean(t_similarity_matrix, dim=-1)

    # calculate student similarity matrix
    s_similarity_matrix = torch.cdist(s_feature, s_feature, p=2)
    s_similarity_matrix = torch.exp(-s_similarity_matrix / T)
    s_similarity_matrix = torch.softmax(s_similarity_matrix, dim=-1)
    # dimension reduction on student similarity matrix
    c_s = torch.mean(s_similarity_matrix, dim=-1)

    # calculate contrastive loss
    loss = (c_t - c_s).pow(2).mean()

    return loss


class StructuredSparsityLoss(torch.nn.Module):
    def __init__(self, alpha):
        super(StructuredSparsityLoss, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * torch.mean(torch.abs(x)) + (1 - self.alpha) * torch.mean(torch.abs(torch.sigmoid(x) - 0.5))
#%%
def make_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_root', type=str, default='data/train', help='architecture name')
    parser.add_argument('--val_root', type=str, default='data/val', help='architecture name')
    parser.add_argument('--n_epoch', type=int, default=1000, help='architecture name')
    parser.add_argument('--batch_size', type=int, default=128, help='architecture name')
    parser.add_argument('--log_freq', type=int, default=10, help='architecture name')
    parser.add_argument('--val_freq', type=int, default=1, help='architecture name')
    parser.add_argument('--save_freq', type=int, default=20, help='architecture name')
    parser.add_argument('--save_dir', type=str, default='output_m_dfscrd', help='architecture name')
    parser.add_argument('--save_dir_dfscrd', type=str, default='output_s_dfscrd', help='architecture name')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--lr_dfscrd', type=float, default=4e-4, help='learning rate')
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

def extract_features(model,layer):
    # 提取中间层特征
    features_extract = nn.Sequential(*list(model.children())[:layer])

    return features_extract

class Trainer:
    def __init__(self, args):
        self.args = args
        self.best_acc = 0.0

        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir_dfscrd, exist_ok=True)


        self.build_model()
        self.build_data()

    def build_model(self):
        self.model_teacher = unet_m(1, 3).to(device)
        self.model_student = unet_s(1, 3).to(device)
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([5, 7, 9], dtype=torch.float32).to(device))
        self.optimizer_teacher = torch.optim.AdamW(self.model_teacher.parameters(), lr=self.args.lr)
        self.optimizer_student = torch.optim.AdamW(self.model_student.parameters(), lr=self.args.lr_dfscrd)

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

    def distillation_dfscrd(self, T, alpha):
        self.best_acc = 0
        train_acc_dfscrd =[]
        train_loss_dfscrd = []
        val_acc_dfscrd = []
        val_loss_dfscrd = []

        for epoch in range(args.n_epoch):

            self.model_teacher.eval()
            self.model_student.train()

            acc_group = np.array([0, 0, 0])
            data_count = np.array([0, 0, 0])
            for step, (data, label) in enumerate(self.data_loader_train):
                data = data.to(device) #[B, 1, N]
                label = label.to(device)

                output_teacher = self.model_teacher(data)
                output_student = self.model_student(data)
                features_teacher = extract_features(self.model_teacher, 1)(data)
                features_student = extract_features(self.model_teacher, 1)(data)

                loss_kl = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output_student/T, dim=1), F.softmax(output_teacher/T, dim=1))*T*T
                # student与teacher feature之间的距离
                loss_feature = CRD(output_teacher, output_student, temperature=T)
                # 计算DFS的loss
                structured_sparsity_loss = StructuredSparsityLoss(alpha=0.2)
                loss_dfs = structured_sparsity_loss(output_student)
                loss_ce = self.criterion(output_student,label)

                loss = alpha*loss_feature + (1-alpha)*loss_dfs

                train_loss_dfscrd.append(loss.item())

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
                    logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            logger.info(f'[{self.args.n_epoch}/{epoch}] <{len(self.data_loader_train)}/{step}>, loss: {loss.item()}')
            macc = acc_group / data_count
            train_acc_dfscrd.append(macc)

            if (epoch + 1) % self.args.val_freq == 0:
                acc, loss = self.val_student()
                val_acc_dfscrd.append(acc)
                val_loss_dfscrd.append(loss.item())
                if np.sum(acc) >= np.sum(self.best_acc):
                    self.best_acc = acc
                    torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir_dfscrd, 'best_model.pth'))

            if (epoch + 1) % self.args.save_freq == 0:
                torch.save(self.model_student.state_dict(), os.path.join(self.args.save_dir_dfscrd, f'{epoch}.pth'))
                # plot loss and accuracy
            plot_loss_and_acc(train_loss_dfscrd, train_acc_dfscrd,val_loss_dfscrd, val_acc_dfscrd,dir=args.save_dir_dfscrd)

if __name__ == '__main__':
    args=make_args()
    trainer=Trainer(args)
    # trainer.train()
    trainer.distillation_dfscrd(T=2, alpha=0.4)