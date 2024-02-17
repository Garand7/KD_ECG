import  torch,os
from unet import unet_m,unet_s


s_model = unet_s(1,3).eval().cuda()
s_model.load_state_dict(torch.load('./output/model_2.pth'),False)

t_model = unet_m(1,3).eval().cuda()
t_model.load_state_dict(torch.load('./output_ks_64_unetms/99.pth'),False)

s_params = list(s_model.parameters())
t_params = list(t_model.parameters())

s_num_params = sum(p.numel() for p in s_params if p.requires_grad)
t_num_params = sum(p.numel() for p in t_params if p.requires_grad)


# 计算模型大小
def get_model_size(model):
    torch.save(model, 'temp.pth')
    size = os.path.getsize('temp.pth') / 1024 / 1024  # MB
    os.remove('temp.pth')
    return size

teacher_size = get_model_size(t_model)
print(f"Teacher model size: {teacher_size} MB")
student_size = get_model_size(s_model)
print(f"Student model size: {student_size} MB")

print(f"教师模型参数量,学生模型参数量:{t_num_params,s_num_params}")
