import torch.nn as nn
import torch

class Classifier(nn.Module):
    # 初始化，传入分类数
    def __init__(self, n_classes=5):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1), # 输出特征形状保持不变
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), # 输出特征形状减半

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(), # 将多维特征图展平为一维向量

            nn.Linear(in_features=16*16*16, out_features=n_classes)
        )

    # 前向传播
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = Classifier(n_classes=5)
    x = torch.randn(32, 3, 64, 64) # 32张64x64的RGB3通道图像
    y = model(x) # 通过卷积层、池化层和全连接层进行前向传播，得到输出是32张图像对应的5个类别的预测结果，即32x5的张量
    print(y.shape)
