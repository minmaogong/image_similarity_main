from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

from common.utils import *
from similarity_config import *
from similarity_data import create_dataset
from similarity_model import ConvEncoder, ConvDecoder
from similarity_engine import evaluate
from similarity_embeddings import get_embedding_collection, search_similar_ids


def test_new_data(encoder, decoder, test_loader, device):
    data_iter = iter(test_loader)
    inputs, targets = next(data_iter)
    inputs = inputs.to(device)

    with torch.no_grad():
        embeddings = encoder(inputs)
        outputs = decoder(embeddings)

    images_numpy = targets.permute(0, 2, 3, 1).numpy()
    outputs_numpy = outputs.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(25, 4), sharex=True, sharey=True)
    for ax_row, imgs in zip(axes, [images_numpy, outputs_numpy]): # 第一行图对应images_numpy， 第二行图对应outputs_numpy
        for ax, img in zip(ax_row, imgs):
            ax.imshow(img)
            ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    seed_everything(SEED)

    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 创建数据集
    _, _, test_dataset = create_dataset()

    # 3. 创建加载器
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE)

    # 4. 加载模型
    encoder = ConvEncoder().to(device)
    decoder = ConvDecoder().to(device)
    encoder.load_state_dict(torch.load(ENCODER_MODEL_NAME, map_location=device))
    decoder.load_state_dict(torch.load(DECODER_MODEL_NAME, map_location=device))

    # 5. 损失函数
    loss_fn = nn.MSELoss()

    # 5. 评估
    test_new_data(encoder, decoder, test_loader, device)
    test_loss = evaluate(encoder, decoder, test_loader, loss_fn, device)

    print(f"测试集误差：{test_loss:.4f}")

    print("===============================================================")

    # 6. 从测试集中获取一张新图片
    image, _ = test_dataset[0]
    print(image.shape)

    # 7. 获取Chroma集合
    collection = get_embedding_collection(encoder)
    print(collection.peek(limit=5))
