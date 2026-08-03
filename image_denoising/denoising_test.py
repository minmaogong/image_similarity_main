import torch
from torch.utils.data import DataLoader

from common.utils import *
from denoising_config import *
from denoising_data import create_datasets
from denoising_engine import test_step
from denoising_model import ConvDenoiser

import matplotlib.pyplot as plt

def test_new_data(model, test_loader, device):
    # 1. 取一个批次的测试图像
    data_iter = iter(test_loader)
    noise_imgs, target_imgs = next(data_iter)

    # 2. 推理预测
    with torch.no_grad():
        noise_imgs = noise_imgs.to(device)

        # 前向传播
        outputs = model(noise_imgs)

    # 3. 转换ndarray 方便画图
    images_numpy = target_imgs.permute(0, 2, 3, 1).numpy()
    noise_imgs_numpy = noise_imgs.cpu().permute(0, 2, 3, 1).numpy()
    denoise_imgs_numpy = outputs.cpu().permute(0, 2, 3, 1).numpy()

    # 4. 画图
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(25, 4), sharex=True, sharey=True)
    for ax_row, imgs in zip(axes, [images_numpy, noise_imgs_numpy, denoise_imgs_numpy]):
        for ax, img in zip(ax_row, imgs):
            ax.imshow(img)
            ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    seed_everything(SEED)

    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 创建数据集
    _, _, test_dataset = create_datasets()
    print("=====================数据集创建完成=====================")

    # 3. 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)
    print("=====================数据加载器创建完成=====================")

    # 4. 加载模型
    model = ConvDenoiser().to(device)
    state_dict = torch.load(DENOISER_MODEL_NAME)
    model.load_state_dict(state_dict)
    print("=====================模型加载完成=====================")

    # 5. 测试
    test_new_data(model, test_loader, device)
    test_loss = test_step(model, test_loader, loss_fn=torch.nn.MSELoss(), device=device)

    print("测试集误差：", test_loss)
