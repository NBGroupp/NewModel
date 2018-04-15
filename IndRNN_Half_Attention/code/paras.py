import math

# maximum absolute value of each recurrent weight
TIME_STEPS = 100
RECURRENT_MAX = pow(2, 1 / TIME_STEPS)

VOCAB_SIZE = 10000 #词典规模
LEARNING_RATE = 0.001 #学习率
LEARNING_RATE_DECAY_FACTOR =  1 #控制学习率下降的参数
KEEP_PROB = 0.6 #节点不Dropout的概率
HIDDEN_SIZE = 128 #词向量维度
PRE_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #上文lstm的隐藏层数目
PRE_CONTEXT_NUM_LAYERS = 2 #上文lstm的深度
FOL_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #下文lstm的隐藏层数目
FOL_CONTEXT_NUM_LAYERS = 2 #下文lstm的深度
VALID_BATCH_SIZE = TEST_BATCH_SIZE = 64 #测试数据batch的大小
TRAIN_BATCH_SIZE = 128 #训练数据batch的大小

DATA_SIZE = 12800
TRAIN_DATA_SIZE = int(DATA_SIZE * 0.6)
VALID_DATA_SIZE = int(DATA_SIZE * 0.2)
TEST_DATA_SIZE = int(DATA_SIZE-VALID_DATA_SIZE-TRAIN_DATA_SIZE)
VALID_STEP_SIZE=math.ceil(VALID_DATA_SIZE / VALID_BATCH_SIZE)
TEST_STEP_SIZE=math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)
TRAIN_STEP_SIZE=math.ceil(TRAIN_DATA_SIZE / TRAIN_BATCH_SIZE)
# TRAIN_STEP_SIZE = 1000
STEP_PRINT = 10
NUM_EPOCH = 1 # 迭代次数




#文件路径
DATA1_PATH = '../model_data/data1.7132608'
DATA2_PATH = '../model_data/data2.7132608'
DATA3_PATH = '../model_data/candidate.7132608'
TARGET_PATH = '../model_data/target.7132608'
VOCAB_PATH = '../model_data/vocab.10000'

CKPT_PATH = '../ckpt/'
MODEL_NAME = 'model.ckpt'
RESULT_PATH = '../results/results.txt'
COST_PATH = '../logs/cost&accuracy_logs'
TEST_RESULT_PATH = '../results/test_results.txt'
