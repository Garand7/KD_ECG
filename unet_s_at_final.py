import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

class ConvSELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvSELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.SELU()
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        self.conv1 = ConvSELU(1, 8, 16, 'same')
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = ConvSELU(8, 16, 16, 'same')
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = ConvSELU(16, 24, 16, 'same')

        self.att2 = Attention(16, 16)  # 注意力模块的通道数需要与对应的卷积层输出匹配
        self.att1 = Attention(8, 8)

        # 注意这里的卷积转置层的输出通道数与注意力模块的输入通道数相匹配
        self.up2 = nn.Sequential(nn.ConvTranspose1d(24, 16, 2, stride=2), ConvSELU(16, 16, 16, 'same'))
        self.up1 = nn.Sequential(nn.ConvTranspose1d(16, 8, 2, stride=2), ConvSELU(8, 8, 16, 'same'))

        self.output = nn.Sequential(
            nn.Conv1d(8, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Encoding path
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))

        # Attention
        x_att2 = self.att2(x2)
        x_att1 = self.att1(x1)

        # Decoding path
        x_up2 = self.up2(x3)
        x_up2 = F.interpolate(x_up2, size=x_att2.size(2))
        x_up2 = x_up2 * x_att2

        x_up1 = self.up1(x_up2)
        x_up1 = F.interpolate(x_up1, size=x_att1.size(2))
        x_up1 = x_up1 * x_att1

        x_out = self.output(x_up1)

        return x_out

def unet_s():
    return AttentionUNet()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet_s().to(device)
    summary(model, (1, 1800))