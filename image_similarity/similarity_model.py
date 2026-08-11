import torch
import torch.nn as nn

# 分别定义编码器类和解码器类
class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        # 通用池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # 输出特征形状为输入的一半

    def forward(self, x):
        x = torch.relu(self.conv1(x)) # 输出：16x64x64
        x = self.pool(x) # 输出： 16x32x32

        x = torch.relu(self.conv2(x)) # 输出：32x32x32
        x = self.pool(x) # 输出： 32x16x16

        x = torch.relu(self.conv3(x)) # 输出：64x16x16
        x = self.pool(x) # 输出： 64x8x8

        x = torch.relu(self.conv4(x)) # 输出：128x8x8
        x = self.pool(x) # 输出： 128x4x4

        x = torch.relu(self.conv5(x)) # 输出：256x4x4
        x = self.pool(x) # 输出： 256x2x2

        x = torch.relu(self.conv6(x)) # 输出：512x2x2
        x = self.pool(x) # 输出： 512x1x1

        # 压缩成向量形式(N, 512)返回
        x = x.squeeze(-1).squeeze(-1) # 输出：512
        return x

class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        # 转置卷积层
        self.conv_t1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1) # 输出特征的形状是输入特征的两倍
        self.conv_t2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t5 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_t6 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        # 恢复4维张量的形状(N, 512, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1) # 输出：512x1x1
        # 转置卷积前向传播
        x = torch.relu(self.conv_t1(x)) # 输出：256x2x2
        x = torch.relu(self.conv_t2(x)) # 输出：128x4x4
        x = torch.relu(self.conv_t3(x)) # 输出：64x8x8
        x = torch.relu(self.conv_t4(x)) # 输出：32x16x16
        x = torch.relu(self.conv_t5(x)) # 输出：16x32x32
        y = torch.sigmoid(self.conv_t6(x)) # 输出：3x64x64，输出范围在(0, 1)

        return y

if __name__ == '__main__':
    input = torch.randn(10, 3, 64, 64)
    encoder = ConvEncoder()
    decoder = ConvDecoder()

    # 编码
    embeddings = encoder(input)
    print("嵌入向量形状：", embeddings.shape) # 输出：[10, 512]

    # 解码
    output = decoder(embeddings)
    print("重构图像形状:", output.shape) # 输出：[10, 3, 64, 64]   =>  [N, C, H, W]
