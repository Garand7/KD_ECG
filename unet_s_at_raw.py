import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn as nn
from torchsummary import summary

class ConvBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p):
        super(ConvBA, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout_p)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.selu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p):
        super(Down, self).__init__()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.convba1 = ConvBA(in_channels, out_channels, kernel_size, stride, dropout_p)
        self.convba2 = ConvBA(out_channels, out_channels, kernel_size, stride, dropout_p)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.convba1(x)
        # x = self.convba2(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2.0, mode='linear')
        self.convba = ConvBA(in_channels, out_channels, kernel_size, stride, dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.convba(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inc = ConvBA(1, 8, kernel_size=16, stride=1, dropout_p=0.2)
        self.down1 = Down(8, 8, kernel_size=16, stride=1, dropout_p=0.2)
        self.up1 = Up(16, 8, kernel_size=16, stride=1, dropout_p=0.2)
        self.outc = nn.Conv1d(8, 3, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.up1(x2, x1)
        x = self.outc(x3)
        x = self.softmax(x)
        return x

def unet_s():
    return MyModel()

model = unet_s()
summary(model.cuda(), input_size=(1, 1800))




