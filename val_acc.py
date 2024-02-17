import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from data import SegData
from sklearn.metrics import f1_score
from loguru import logger
from unet_bk import unet_s, unet_m, unet_l
from unet import unet_s, unet_m, unet_l
# from unet_s_final import unet_s
# from unet_s_at_exp3 import unet_s
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

val_root = "data/val"
dataset_val = SegData(val_root)
data_loader_val = DataLoader(dataset=dataset_val, batch_size=64, shuffle=False, pin_memory=True, num_workers=8)


model = unet_s(1,3).eval().cuda()
# model = unet_s().eval().cuda()
model.load_state_dict(torch.load('./output_s_crd_self_225/best_model.pth'),False)

# pred = pred.cpu().detach().numpy()
# #%%
# model.eval()
#
# acc_group = np.array([0, 0, 0])
# data_count = np.array([0, 0, 0])
# for data, label in data_loader_val:
#     data = data.to(device)
#     label = label.to(device)
#
#     data_count += len(label)
#
#     output = model(data)
#     pred = torch.argmax(output, dim=1)
#
#     for i in range(3):
#         idx_cls = label == i
#         data_count[i] += idx_cls.sum()
#         acc_group[i] += (label[idx_cls] == pred[idx_cls]).sum().item()
#
# m_acc = acc_group / data_count
# pred = pred.reshape(-1)
# label = label.reshape(-1)
# logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)}')
# f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='micro')
# logger.info(f'f1 score: {f1:.3}')
#%%
def calculate():
    # calculate model size and parameter count
    num_params = sum(p.numel() for p in model.parameters())
    model_size = num_params * 4 / (1024 * 1024)
    logger.info(f"Number of parameters: {num_params}")
    logger.info(f"Model size: {model_size:.2f} MB")

def val():
    model.eval()

    acc_group = np.array([0, 0, 0])
    data_count = np.array([0, 0, 0])
    for data, label in data_loader_val:
        data = data.to(device)
        label = label.to(device)

        data_count += len(label)

        output = model(data)
        pred = torch.argmax(output, dim=1)

        for i in range(3):
            idx_cls = label == i
            data_count[i] += idx_cls.sum()
            acc_group[i] += (label[idx_cls] == pred[idx_cls]).sum().item()

    m_acc = acc_group / data_count
    q = np.sum(m_acc)
    ave_acc = q/3
    pred = pred.reshape(-1)
    label = label.reshape(-1)
    logger.info(f'm_acc: {", ".join(f"{x:.3}" for x in m_acc)}')
    f1 = f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
    logger.info(f'f1 score: {f1:.3}')
    logger.info(f'ave_acc: {ave_acc:.3}')
    return m_acc

if __name__ == '__main__':
    val()
    calculate()