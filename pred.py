import torch
import os
import numpy as np
from matplotlib.lines import Line2D

from unet import unet_s, unet_m, unet_l
# from unet_s_at_exp import unet_s
# from sklearn import preprocessing as prep
import matplotlib.pyplot as plt
import time
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

val_sig_path = './data/val/sigs/'
val_label_path = './data/val/labels/'

sig_files = os.listdir(val_sig_path)
label_files = os.listdir(val_label_path)

select = np.random.choice(sig_files, 1)[0]
# select = sig_files[0]
a_sig = np.load(val_sig_path+select)
a_seg = np.load(val_label_path+select)
#%%
model = unet_m(1,3).eval().cuda()
model.load_state_dict(torch.load('./output_m_crd_1d/best_model.pth'),False)
# model = torch.load('./output/199.ckpt')
# #%%
data = torch.from_numpy(a_sig).float().unsqueeze(0).unsqueeze(0).cuda()
# #%%
out = model(data)
#
# #%%
pred=torch.argmax(out, dim=1)
#%%
pred = pred.cpu().detach().numpy()
#%%
# out = out.cpu().detach().numpy()
# #%%
#
# tic = time.time()
#
# toc = time.time()
# #%%
# a = torch.from_numpy(a_sig).float()
# a = a.unsqueeze(0)
# a = a.cpu().detach().numpy()
# b = np.row_stack((a,pred[0,:]))

#%%
# print('Elapsed time: '+str(toc-tic)+' seconds.')
#
# plt.grid(True)
#
# x = [i for i in range(0, 1800)]
# # 创建一个字典来存储每个标签对应的颜色
# color_dict = {0: 'blue', 1: 'green', 2: 'red'}#0 背景 1 normal 2 PVC
# legend_dict = {'Background': 'blue', 'Normal': 'green', 'PVC': 'red'}
# # 绘制散点图
# for i in range(1800):
#     plt.scatter(x[i], a_sig[i], c=color_dict[pred[0,i]])
#
# legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
#                           markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
# plt.legend(handles=legend_elements, loc='lower right')
# plt.show()
plt.grid(True)

x = [i for i in range(0, 1800)]
# 创建一个字典来存储每个标签对应的颜色
color_dict = {0: 'blue', 1: 'green', 2: 'red'}#0 背景 1 normal 2 PVC
legend_dict = {'Background': 'blue', 'Normal': 'green', 'PVC': 'red'}
# 绘制折线图
for label, color in color_dict.items():
    y = [a_sig[i] if pred[0,i]==label else None for i in range(1800)]
    plt.plot(x, y, color=color)

legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                          markerfacecolor=value, markersize=10) for key, value in legend_dict.items()]
plt.legend(handles=legend_elements, loc='lower right')
plt.savefig("pred_01.png")
plt.show()

#%%
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# GPU预热
for _ in range(50):
    _ = model(data)

iterations = 300   # 重复计算的轮次
# 测速
times = torch.zeros(iterations)     # 存储每轮iteration的时间
with torch.no_grad():
    for iter in range(iterations):
        starter.record()
        _ = model(data)
        ender.record()
        # 同步GPU时间
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 计算时间
        times[iter] = curr_time
        # print(curr_time)

mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))



