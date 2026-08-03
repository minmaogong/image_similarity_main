import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from denoising_config import *
from denoising_model import ConvDenoiser
from common.utils import *
from denoising_data import *
from denoising_engine import *
from tqdm import tqdm # 进度条库，用于显示训练进度

if __name__ == "__main__":
    seed_everything(SEED) # 设定随机种子，确保每次运行结果一致，消除随机性

    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建数据集
    train_dataset, val_dataset, _ = create_datasets()
    print("=====================数据集创建完成=====================")

    # 3. 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    print("=====================数据加载器创建完成=====================")

    # 4. 创建模型
    model = ConvDenoiser().to(device)

    # 5. 损失函数
    loss_fn = nn.MSELoss()

    # 6. 优化器
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # 7. 训练核心流程
    print("=====================训练开始=====================")
    min_val_loss = float("inf")
    for epoch in tqdm(range(EPOCHS)):
        # 训练
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        # 验证
        val_loss = val_step(model, val_loader, loss_fn, device)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # 判断如果验证损失减小，就保存模型
        if val_loss < min_val_loss:
            print("验证集损失减小，保存模型...")
            min_val_loss = val_loss
            torch.save(model.state_dict(), DENOISER_MODEL_NAME)
        else:
            print("验证集损失没有减小，继续训练...")

    print("=====================训练结束=====================")
