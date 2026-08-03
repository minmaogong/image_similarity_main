import torch
from torch import nn, optim

from denoising_config import *
from denoising_model import ConvDenoiser

if __name__ == "__main__":
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvDenoiser().to(device)

    # 损失函数
    loss_fn = nn.MSELoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for noise_img, target_img in train_loader:
            noise_img = noise_img.to(device)
            target_img = target_img.to(device)
            # 前向传播，得到输出值
            output = model(noise_img)
            # 计算损失
            loss = loss_fn(output, target_img)
            # 反向传播，计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            # 梯度清零
            optimizer.zero_grad()

            total_loss += loss.item()

        # 本轮训练结束，计算平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
