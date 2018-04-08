VOCAB_SIZE = 10000

# 训练数据参数
DATA_SIZE = 42
TRAIN_DATA_SIZE = int(DATA_SIZE * 0.6)

# 测试数据参数
# DATA_SIZE = 42
# TEST_DATA_SIZE = 42


# 文件路径
DATA0_PATH = '../model_data/data0.'+str(DATA_SIZE)
DATA1_PATH = '../model_data/data1.'+str(DATA_SIZE)
DATA2_PATH = '../model_data/data2.'+str(DATA_SIZE)
DATA3_PATH = '../model_data/candidate.'+str(DATA_SIZE)
TARGET_PATH = '../model_data/target.'+str(DATA_SIZE)
VOCAB_PATH = '../model_data/vocab.10000'

COUNT_SAVE_PATH = '../count.dict'
TEST_RESULT_PATH = '../results/test_results.txt'

STEP_PRINT = 1000 # 输出步频

TP = FP = TN = FN = P = N = 0
TPR = TPW =0
