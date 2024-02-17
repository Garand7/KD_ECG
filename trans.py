import torch
import onnx,os
from unet import unet_s,unet_m

model = unet_s(1,3).cuda()

model.load_state_dict(torch.load("./output_s_crd_1d/best_model.pth"))
# # 加载 PyTorch 模型
# model = torch.load("./output_s_crd_1d/999.pth")
#%%
# 创建一个 PyTorch 跟踪器对象
input_shape = (1800,1,3)
input_example = torch.randn(input_shape).cuda()


# 将模型转换为 ONNX 格式
output_path = "model.onnx"
#%%
torch.onnx.export(model, input_example, output_path)

# 加载 ONNX 模型并检查其是否有效
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)