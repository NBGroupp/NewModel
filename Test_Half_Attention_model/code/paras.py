import math

VOCAB_SIZE = 10000 #词典规模

LEARNING_RATE = 0.001 #学习率
LEARNING_RATE_DECAY_FACTOR =  1 #控制学习率下降的参数

HIDDEN_SIZE = 128 #词向量维度
PRE_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #上文lstm的隐藏层数目
PRE_CONTEXT_NUM_LAYERS = 1 #上文lstm的深度
FOL_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #下文lstm的隐藏层数目
FOL_CONTEXT_NUM_LAYERS= 1 #下文lstm的深度


VALID_BATCH_SIZE = TEST_BATCH_SIZE = 100 #测试数据batch的大小

DATA_SIZE = 24378315
TEST_DATA_SIZE = 100000

TEST_STEP_SIZE=math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)

#文件路径 

DATA0_PATH = '../model_data/error_origins.'+str(DATA_SIZE)
DATA1_PATH = '../model_data/data1.'+str(DATA_SIZE)
DATA2_PATH = '../model_data/data2.'+str(DATA_SIZE)
DATA3_PATH = '../model_data/nears.'+str(DATA_SIZE)
TARGET_PATH = '../model_data/target.'+str(DATA_SIZE)
VOCAB_PATH = '../model_data/vocab.10000'

CKPT_PATH = '../ckpt/'
MODEL_NAME = 'model.ckpt'
RESULT_PATH = '../results/results.txt'
COST_PATH = '../logs/cost&accuracy_logs'
TEST_RESULT_PATH = '../results/test_results.txt'

STEP_PRINT = 1000 # 输出步频
PROOFREAD_BIAS = 0.01 # 校对阈值

TP = FP = TN = FN = P = N = 0
TPW = TPR = 0
