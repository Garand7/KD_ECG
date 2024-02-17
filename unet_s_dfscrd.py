import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        # Simplify the attention module to use fewer channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply attention to each channel independently and then expand it back to the input size
        attention = self.conv(x)
        return x * attention

class SimplifiedAttentionUNet(nn.Module):
    def __init__(self):
        super(SimplifiedAttentionUNet, self).__init__()

        # Reduce the number of channels in each layer to minimize the model size
        self.conv1 = nn.Conv1d(1, 8, kernel_size=16, padding='same')
        self.pool1 = nn.MaxPool1d(2)
        self.att1 = Attention(8)

        self.conv2 = nn.Conv1d(8, 16, kernel_size=16, padding='same')
        self.att2 = Attention(16)

        # Use a single upsample layer for simplicity
        self.up = nn.Upsample(scale_factor=2)

        # Output layer
        self.out_conv = nn.Conv1d(16, 3, kernel_size=1)
        self.out_act = nn.Softmax(dim=1)

    def forward(self, x):
        # Encoding path with attention
        x1 = self.conv1(x)
        x1_att = self.att1(x1)

        x2 = self.pool1(x1_att)
        x2 = self.conv2(x2)
        x2_att = self.att2(x2)

        # Decoding path
        x_up = self.up(x2_att)

        # Output layer
        x_out = self.out_act(self.out_conv(x_up))

        return x_out

def unet_s():
    return SimplifiedAttentionUNet()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = unet_s().to(device)
    # 输出模型总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    summary(model,(1,1800))