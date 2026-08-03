# 数据预处理
IMG_PATH = "../common/dataset/"
IMG_H = 64
IMG_W = 64

# 随机性相关配置
SEED = 42
TRAIN_RATIO = 0.7 # 训练集比例
VAL_RATIO = 0.15 # 验证集比例
TEST_RATIO = 0.15 # 测试集比例
NOISE_FACTOR = 0.5 # 噪声因子

# 超参数
LEARNING_RATE = 1e-3
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
EPOCHS = 30

# 项目配置
PACKAGEA_NAME = "image_denoising"
DENOISER_MODEL_NAME = "denoiser.pt"
