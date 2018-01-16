import math

VOCAB_SIZE = 112801 #词典规模
MAX_TEXT_LENGTH = 50 #最长文本长度

LEARNING_RATE = 0.001 #学习率
LEARNING_RATE_DECAY_FACTOR =  1 #控制学习率下降的参数
KEEP_PROB = 0.75 #节点不Dropout的概率
# MAX_GRAD_NORM = 5 #用于控制梯度膨胀的参数

HIDDEN_SIZE = 128 #词向量维度
PRE_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #上文lstm的隐藏层数目
PRE_CONTEXT_NUM_LAYERS = 1 #上文lstm的深度
FOL_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #下文lstm的隐藏层数目
FOL_CONTEXT_NUM_LAYERS= 1 #下文lstm的深度


VALID_BATCH_SIZE = TEST_BATCH_SIZE = 32 #测试数据batch的大小
#VALID_NUM_STEP = TEST_NUM_STEP = 1 #测试数据截断长度
TRAIN_BATCH_SIZE = 128 #训练数据batch的大小
#TRAIN_NUM_STEP =35 #训练数据截断长度
GRAM = 3 #LSTM的时间维度

DATA_SIZE = 3698375
TRAIN_DATA_SIZE = int(DATA_SIZE * 0.6)
VALID_DATA_SIZE = int(DATA_SIZE * 0.2)
TEST_DATA_SIZE = int(DATA_SIZE-VALID_DATA_SIZE-TRAIN_DATA_SIZE)

VALID_EPOCH_SIZE=math.ceil(VALID_DATA_SIZE / VALID_BATCH_SIZE)
TEST_EPOCH_SIZE=math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)
TRAIN_EPOCH_SIZE=math.ceil(TRAIN_DATA_SIZE / TRAIN_BATCH_SIZE)
NUM_EPOCH = 100 # 迭代次数

#文件路径
DATA1_PATH = '../model_data/data1.18876'
DATA2_PATH = '../model_data/data2.18876'
TARGET_PATH = '../model_data/target.18876'
VOCAB_PATH = '../model_data/vocab.100000'

CKPT_PATH = '../ckpt/'
MODEL_NAME = 'model.ckpt'
RESULT_PATH = '../results/results.txt'
COST_PATH = '../logs/cost_logs'
TEST_RESULT_PATH = '../results/test_results.txt'
