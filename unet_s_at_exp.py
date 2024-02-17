import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class ConvDropoutSELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvDropoutSELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.2),
            nn.SELU()
        )

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()

        self.conv1 = ConvDropoutSELU(1, 8, 16, 'same')
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = ConvDropoutSELU(8, 8, 16, 'same')
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = ConvDropoutSELU(8, 16, 16, 'same')
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = ConvDropoutSELU(16, 32, 16, 'same')
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = ConvDropoutSELU(32, 32, 16, 'same')
        self.pool5 = nn.MaxPool1d(2)
        self.conv6 = ConvDropoutSELU(32, 32, 16, 'same')
        self.conv7 = ConvDropoutSELU(32, 32, 16, 'same')

        self.att4 = Attention(32, 32)
        self.att3 = Attention(16, 16)
        self.att2 = Attention(8, 8)
        self.att1 = Attention(8, 8)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), ConvDropoutSELU(32, 32, 16, 'same'))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), ConvDropoutSELU(32, 16, 16, 'same'))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), ConvDropoutSELU(16, 8, 16, 'same'))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), ConvDropoutSELU(8, 8, 16, 'same'))

        self.output = nn.Sequential(
            nn.Conv1d(8, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        x6 = self.conv6(self.pool5(x5))
        x7 = self.conv7(x6)

        x_att4 = self.att4(x4)
        x_exp = self.up4(x7)
        # print("x_att4 shape:", x_att4.shape)
        # print("x_exp shape:", x_exp.shape)

        # Match the shapes of x_exp and x_att4 by padding or cropping x_exp
        diff = x_att4.size(2) - x_exp.size(2)
        if diff > 0:
            # Add padding to x_exp
            x_exp = F.pad(x_exp, (0, diff))
        elif diff < 0:
            # Crop x_exp
            x_exp = x_exp[:, :, :diff]

        x_up4 = x_exp * x_att4

        # x_up4 = self.up4(x7) * x_att4

        x_att3 = self.att3(x3)
        x_up3 = self.up3(x_up4 + x4) * x_att3
        x_att2 = self.att2(x2)
        x_up2 = self.up2(x_up3 + x3) * x_att2
        x_att1 = self.att1(x1)
        x_up1 = self.up1(x_up2 + x2) * x_att1

        x_out = self.output(x_up1)
        return x_out

def unet_s():

    return AttentionUNet()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet_s().to(device)
    # print(model)
    summary(model, (1, 1800))