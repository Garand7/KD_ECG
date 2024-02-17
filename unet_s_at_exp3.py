import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# 注意力机制
class SqueezeAndExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeAndExcitation, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.SELU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ConvBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, padding='same', dropout_rate=0.2):
        super(ConvBA, self).__init__()

        if padding == 'same':
            padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout_rate),
            nn.SELU(inplace=True)
        )
        self.se = SqueezeAndExcitation(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(Down, self).__init__()

        self.down = nn.Sequential(
            nn.MaxPool1d(2),
            ConvBA(in_channels, out_channels, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, dropout_rate=0.2):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
            self.conv = ConvBA(in_channels, out_channels, dropout_rate=dropout_rate)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2,padding='same')
            self.conv = ConvBA(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, (diff_y // 2, diff_y - diff_y // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = ConvBA(1, 8)
        self.down_layers = nn.ModuleList([
            Down(8, 16),
            Down(16, 32),
            Down(32, 32)
        ])
        self.mid_layers = nn.Sequential(
            ConvBA(32, 32)
        )
        self.up_layers = nn.ModuleList([
            Up(64, 16),
            Up(32, 8),
            Up(16, 8)
        ])
        self.outc = nn.Conv1d(8, 3, kernel_size=1, stride=1, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down_layers[0](x1)
        x3 = self.down_layers[1](x2)
        x4 = self.down_layers[2](x3)
        xm = self.mid_layers(x4)
        x5 = self.up_layers[0](xm, x3)
        x6 = self.up_layers[1](x5, x2)
        x7 = self.up_layers[2](x6, x1)
        out = self.outc(x7)
        out = self.softmax(out)
        return out


def unet_s():

    return UNet()

model = unet_s()
summary(model.cuda(), input_size=(1, 1800))