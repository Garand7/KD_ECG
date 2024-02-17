""" Parts of the U-Net model """
from torchsummary import summary

"""https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBA(nn.Sequential):
    def __init__(self, in_channels, out_channels, n_conv=2):
        layers=[
            nn.Conv1d(in_channels, out_channels, kernel_size=16,padding='same'), #16
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.2),
            nn.SELU(inplace=True),  # 高级激活函数
        ]
        # for i in range(n_conv-1):
        #     layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=64))
        #     layers.append(nn.BatchNorm1d(out_channels))
        #     layers.append(nn.Dropout(0.2))
        #     layers.append(nn.SELU(inplace=True))
        super().__init__(*layers)

class Down(nn.Sequential):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, n_conv=1):
        super().__init__(
            nn.MaxPool1d(2),
            ConvBA(in_channels, out_channels, n_conv)
        )

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, n_conv=1):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2,padding='same')

        self.conv = ConvBA(in_channels, out_channels, n_conv=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffX = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,padding='same')

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch, n_classes, blocks, bilinear=True, n_conv=2):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ConvBA(in_ch, blocks[0])
        down_layers=[Down(blocks[i], blocks[i+1], n_conv=n_conv) for i in range(len(blocks)-1)]
        down_layers.append(Down(blocks[-1], blocks[-1], n_conv=n_conv))
        self.down_layers = nn.ModuleList(down_layers)

        self.mid_layers = nn.Sequential(
            ConvBA(blocks[-1], blocks[-1])
        )

        up_layers = [Up(blocks[i]*2, blocks[i-1], n_conv=n_conv) for i in range(len(blocks)-1, 0, -1)]
        up_layers.append(Up(blocks[0]*2, blocks[0], n_conv=n_conv))
        self.up_layers = nn.ModuleList(up_layers)

        self.outc = nn.Conv1d(blocks[0], n_classes, kernel_size=1,padding='same')
        self.softmax = nn.Softmax(dim=1)

    def get_intermediate_layer(model, layer_name, input_data):
        """
        Get the intermediate layer output of a model given the layer name and input data.
        """
        intermediate_layer_model = torch.nn.Sequential(
            *list(model.children())[:list(model.named_children()).index((layer_name, model._modules[layer_name])) + 1])
        with torch.no_grad():
            intermediate_output = intermediate_layer_model(input_data)
        return intermediate_output

    def forward(self, x):
        x = self.inc(x)

        x_list = [x]
        for layer in self.down_layers:
            x=layer(x)
            x_list.append(x)

        x=self.mid_layers(x)

        for layer, x_d in zip(self.up_layers, x_list[-2::-1]):
            x=layer(x, x_d)

        logits = self.outc(x)
        logits = self.softmax(logits)
        return logits

def unet_s(in_ch, n_classes):
    #blocks=[10,10,10,16,20,20,32,32]
    blocks=[16] #4
    return UNet(in_ch, n_classes, blocks)

def unet_m(in_ch, n_classes):
    blocks=[8,16,32,64,128,256]
    return UNet(in_ch, n_classes, blocks)

def unet_l(in_ch, n_classes):
    blocks=[16,32,64,128,256,512]
    return UNet(in_ch, n_classes, blocks, n_conv=3)

def unet_xl(in_ch, n_classes):
    blocks=[64,128,256,512,1024]
    return UNet(in_ch, n_classes, blocks, n_conv=6)


if __name__ == '__main__':
    print('\nSummarize the model:\n')
    model = unet_s(1,3)
    model2 = unet_m(1,3)
    #model.summary()
    print('\nEnd for summary.\n')
    summary(model.cuda(),input_size=(1,1800))
    # summary(model2.cuda(),input_size=(1,1800))