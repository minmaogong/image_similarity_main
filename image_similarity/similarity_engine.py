import torch
from tqdm import tqdm

# 训练一个轮次
def train_one_epoch(encoder, decoder, train_loader, loss_fn, optimizer, device):
    # 训练
    encoder.train()
    decoder.train()

    total_loss = 0
    total_num = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # 前向传播
        embeddings = encoder(inputs)
        outputs = decoder(embeddings)
        # 计算损失
        loss = loss_fn(outputs, targets)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()

        total_num += inputs.shape[0]
        total_loss += loss.item() * inputs.shape[0]
    # 本轮训练结束，返回平均损失
    return total_loss / total_num

# 验证
def validate(encoder, decoder, val_loader, loss_fn, device):
    # 验证
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播
            embeddings = encoder(inputs)
            outputs = decoder(embeddings)
            # 计算损失
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]

    # 验证结束，返回平均损失
    return total_loss / total_num

# 评估
def evaluate(encoder, decoder, test_loader, loss_fn, device):
    # 测试
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # 前向传播
            embeddings = encoder(inputs)
            outputs = decoder(embeddings)
            # 计算损失
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]

    # 测试结束，返回平均损失
    return total_loss / total_num

