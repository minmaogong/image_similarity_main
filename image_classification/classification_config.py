# 数据预处理
IMG_PATH = "../common/dataset/"
LABELS_PATH = "../common/fashion-labels.csv"
IMG_H = 64
IMG_W = 64

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.7 # 训练集比例
VAL_RATIO = 0.15 # 验证集比例
TEST_RATIO = 0.15 # 测试集比例

# 超参数
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 20

# 项目配置
PACKAGEA_NAME = "image_classification"
CLASSIFIER_MODEL_NAME = "classifier.pt"

# 定义标签和中文名称映射关系
classification_names = {
    0: '上衣',
    1: '鞋',
    2: '包',
    3: '下装',
    4: '手表'
}
