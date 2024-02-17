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

        self.conv2 = ConvDropoutSELU(8, 10, 16, 'same')
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = ConvDropoutSELU(10, 16, 16, 'same')
        self.pool3 = nn.MaxPool1d(2)

        self.conv4 = ConvDropoutSELU(16, 24, 16, 'same')
        self.pool4 = nn.MaxPool1d(2)

        self.conv5 = ConvDropoutSELU(24, 28, 16, 'same')

        self.att4 = Attention(24, 24)
        self.att3 = Attention(16, 16)
        self.att2 = Attention(10, 10)

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2), ConvDropoutSELU(28, 24, 16, 'same'))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2), ConvDropoutSELU(24, 16, 16, 'same'))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2), ConvDropoutSELU(16, 10, 16, 'same'))
        self.up0 = nn.Sequential(nn.Upsample(scale_factor=2), ConvDropoutSELU(10, 10, 16, 'same'))

        self.output = nn.Sequential(
            nn.Conv1d(10, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # 编码路径
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))

        # 注意力
        x_att4 = self.att4(x4)
        x_att3 = self.att3(x3)
        x_att2 = self.att2(x2)

        # 解码路径
        x_up4 = self.up4(x5)
        x_up4 = F.interpolate(x_up4, size=225)
        x_up4 = x_up4 * x_att4

        x_up3 = self.up3(x_up4)
        x_up3 = x_up3 * x_att3

        x_up2 = self.up2(x_up3)
        x_up2 = x_up2 * x_att2
        x_up2 = self.up0(x_up2)

        x_out = self.output(x_up2)

        return x_out

def unet_s():

    return AttentionUNet()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet_s().to(device)
    # print(model)
    summary(model, (1, 1800))

# 这段代码定义了一个基于U-Net架构的卷积神经网络，该网络用于处理一维的数据。它通过引入注意力机制来增强网络性能。以下是网络各部分的详细解释：
#
# 注意力模块（Attention）
# 这个子模块定义了一个注意力机制，它接收特定通道数的输入，通过一个1x1卷积层，批量归一化（BatchNorm1d），然后通过一个Sigmoid激活函数来生成注意力权重。这个注意力权重会用于后续的特征地图，以突出重要的特征并抑制不重要的特征。
#
# 卷积-丢弃-SELU模块（ConvDropoutSELU）
# 这是一个标准的卷积块，它包括卷积层、批量归一化、丢弃层以及SELU激活函数。SELU是一种自归一化激活函数，它可以在训练深层网络时保持梯度的稳定。
#
# 注意力U-Net模型（AttentionUNet）
# 这是主模型，它使用了多个ConvDropoutSELU模块来构建U-Net架构的编码器和解码器路径，并在解码器路径中利用了注意力模块来提升特征的表达。
#
# 在编码阶段，输入数据通过一系列的卷积和池化操作，逐步减小特征地图的尺寸，同时增加通道数，这可以帮助网络捕获更高层次的特征。
#
# 在解码阶段，特征地图被上采样（通过nn.Upsample）和卷积，逐渐恢复到原始输入的尺寸。在每次上采样之后，都会应用注意力模块的输出来加权特征地图，这有助于网络集中于更加相关的特征。
#
# 在最后，通过一个卷积层将通道数减少到3（假设是为了三分类问题），然后应用Softmax激活函数来获取分类的概率分布。