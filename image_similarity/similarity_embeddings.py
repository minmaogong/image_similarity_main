import os
from math import ceil

import chromadb
import torch
import numpy as np
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Images
from PIL import Image
import torchvision.transforms as T

from common.utils import sorted_alphanum
from similarity_config import *

# 自定义嵌入函数
class ImageEmbeddingFunction(EmbeddingFunction[Images]):
    # 初始化，传入自己的嵌入模型（Encoder）
    def __init__(self, model) -> None:
        self.model = model.to('cpu')

    # 调用方法
    def __call__(self, input: Images) -> Embeddings:
        # 将输入图像转换为 Tensor
        input_tensor = torch.tensor(np.array(input))
        # 前向传播，得到模型的输出
        with torch.no_grad():
            output = self.model(input_tensor)
        # 将输出转换为 ndarray 返回
        return output.numpy()


# 加载所有图片，返回一个字典{id, image}
def get_id2images(image_dir, transform):
    id2images = {}
    # 读取目录下所有图片文件名
    image_names = sorted_alphanum(os.listdir(image_dir))
    # 遍历每个文件名，打开图片进行转换
    for i, image_name in enumerate(image_names):
        # 1.1 构建图片的完整访问路径
        image_path = os.path.join(image_dir, image_name)
        # 1.2 打开图片
        image = Image.open(image_path).convert('RGB')
        # 1.3 应用转换操作，得到Tensor
        img_tensor = transform(image)
        # 1.4 转换成ndarray，保存到字典
        id2images[str(i)] = img_tensor.numpy()

    return id2images

# 获取Chroma集合
def get_embedding_collection(encoder):
    # 2.1 创建客户端
    path = os.path.join('..', PACKAGE_NAME, CHROMA_BACKEND_PATH)
    print("path", path)
    client = chromadb.PersistentClient(path=path)
    # 2.2 创建集合
    collection = client.get_or_create_collection(
        name=IMAGE_COLLECTION_NAME,
        embedding_function=ImageEmbeddingFunction(encoder)
    )
    return collection

# 生成所有图像的嵌入向量（预处理）
def create_embeddings(encoder):
    # 1. 加载所有图片
    transform = T.Compose([
        T.Resize((IMG_H, IMG_W)),
        T.ToTensor(),
    ])
    print("正在加载所有图片...")
    id2images = get_id2images(IMG_PATH, transform=transform)
    print("图片加载完毕！")

    ids = list(id2images.keys())
    imgs = list(id2images.values())

    # 2. 获取Chroma的集合
    collection = get_embedding_collection(encoder)

    # 3. 执行写入Chroma的操作
    print("开始写入Chroma数据库...")
    # 分批写入
    batchs = ceil(len(ids) / CHROMA_INSERT_BATCH_SIZE)
    for i in range(batchs):
        start = min(i * CHROMA_INSERT_BATCH_SIZE, len(ids))
        end = min((i+1) * CHROMA_INSERT_BATCH_SIZE, len(ids))
        collection.upsert(
            ids=ids[start:end],
            images=imgs[start:end],
        )
    print("写入Chroma数据库完毕！")


# 相似图片搜索
def search_similar_ids(collections, image, cnt):
    pass

