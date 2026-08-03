from torch import nn
import torch

# 自定义神经网络类: 基于CNN的去噪器
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        # 编码器
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1) # 输出特征形状不变
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 解码器
        # 转置卷积层
        self.conv_t1 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1) # 输出特征形状为输入的两倍 也可以设置成 kernel_size=2, stride=2, padding=0, output_padding=0, 输出特征同样为输入特征的两倍
        self.conv_t2 = nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t3 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # 输出普通卷积层
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)

    # 前向传播
    def forward(self, x):
        # 编码
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # 解码
        x = torch.relu(self.conv_t1(x))

        x = torch.relu(self.conv_t2(x))

        x = torch.relu(self.conv_t3(x))

        # 输出
        y = torch.sigmoid(self.conv_out(x))

        return y

if __name__ == '__main__':
    input = torch.randn(12, 3, 64, 64)
    model = ConvDenoiser()
    output = model(input)
    print(output.shape)
