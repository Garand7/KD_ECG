import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=16, stride=1, padding='same', dilation=1,
                 groups=1, bias=True, padding_mode='zeros'):
        super(ConvBA, self).__init__()

        # if padding == 'same':
        #     padding = (kernel_size - 1) // 2
        # else:
        #     padding = 0

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
                              padding_mode)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.2)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.selu(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv = ConvBA(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = ConvBA(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Add interpolation to ensure the shapes match
        diff = x2.size()[2] - x1.size()[2]
        if diff > 0:
            x1 = F.interpolate(x1, size=(x2.size()[2]), mode='linear', align_corners=True)
        elif diff < 0:
            x2 = F.interpolate(x2, size=(x1.size()[2]), mode='linear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Ensure the shapes of g1 and x1 match
        diff = x1.size()[2] - g1.size()[2]
        if diff > 0:
            g1 = F.interpolate(g1, size=(x1.size()[2]), mode='linear', align_corners=True)
        elif diff < 0:
            x1 = F.interpolate(x1, size=(g1.size()[2]), mode='linear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.inc = ConvBA(1, 8)
        self.down1 = Down(8, 8)
        self.mid = ConvBA(8, 8)
        self.attention1 = AttentionBlock(F_g=8, F_l=8, F_int=4)
        self.up1 = Up(16, 8)
        self.outc = nn.Conv1d(8, 3,kernel_size=1, stride=1, padding='same')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        xm = self.mid(x2)
        xm = self.attention1(xm, x2)
        x = self.up1(xm, x1)
        x = self.outc(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    summary(model, (1, 1800))