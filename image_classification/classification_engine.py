import torch
from tqdm import tqdm

# 训练一个轮次
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    # 训练
    model.train()
    total_loss = 0
    total_num = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = loss_fn(outputs, labels)
        # 前向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 累加损失
        total_num += images.shape[0]
        total_loss += loss.item()*images.shape[0]
    # 本轮验证结束，返回平均损失
    return total_loss / total_num

# 验证
def validate(model, val_loader, loss_fn, device):
    # 验证
    model.eval()
    total_loss = 0
    total_num = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = loss_fn(outputs, labels)

            # 累加损失
            total_num += images.shape[0]
            total_loss += loss.item() * images.shape[0]
    # 返回平均损失
    return total_loss / total_num

# 测试评估
def evaluate(model, test_loader, device):
    # 测试
    model.eval()
    total_num = 0
    test_acc_num = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='测试'):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            # 得到预测分类标签
            pred_labels = outputs.argmax(dim=-1)
            # 累加预测准确的个数
            total_num += images.shape[0]
            test_acc_num += pred_labels.eq(labels).sum().item()

    # 计算准确率
    return test_acc_num / total_num

