import torch
from tqdm import tqdm


# 训练一个轮次
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
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
    return total_loss / len(train_loader)

# 验证一个轮次
def val_step(model, val_loader, loss_fn, device):
    # 验证
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noise_img, target_img in val_loader:
            noise_img = noise_img.to(device)
            target_img = target_img.to(device)
            # 前向传播，得到输出值
            output = model(noise_img)
            # 计算损失
            loss = loss_fn(output, target_img)

            total_loss += loss.item()
    # 计算平均损失
    return total_loss / len(val_loader)

# 测试一个轮次
def test_step(model, test_loader, loss_fn, device):
    # 测试
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for noise_img, target_img in tqdm(test_loader, desc="测试"):
            noise_img = noise_img.to(device)
            target_img = target_img.to(device)
            # 前向传播，得到输出值
            output = model(noise_img)
            # 计算损失
            loss = loss_fn(output, target_img)

            total_loss += loss.item()
    # 计算平均损失
    return total_loss / len(test_loader)
