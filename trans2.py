import torch
from caffe2.python.onnx import backend as caffe2_backend
from caffe2.python.onnx.frontend import convert
# Load the PyTorch model
import torch
import onnx
from unet import unet_s,unet_m

model = unet_s(1,3).cuda()

model.load_state_dict(torch.load("./output_s_crd_1d/best_model.pth"))
#%%
# Convert the PyTorch model to ONNX format
input_shape = (1,3,1800)
input_example = torch.randn(input_shape)
output_path = 'model.onnx'
# torch.onnx.export(model, input_example, output_path)
#%%
# Convert the ONNX model to Caffe2 format
caffe2_net = convert(output_path)

# Save the Caffe2 model
with open('model.pb', 'wb') as f:
    f.write(caffe2_net.SerializeToString())