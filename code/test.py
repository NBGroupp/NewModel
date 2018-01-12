import tensorflow as tf
import os
import numpy as np
from model import *

TEST_BATCH_SIZE = 20 #测试数据batch的大小
#TEST_NUM_STEP = 1 #测试数据截断长度
GRAM = 3 #LSTM的时间维度

DATA_SIZE = 18876
TRAIN_DATA_SIZE = DATA_SIZE * 0.6
TEST_DATA_SIZE = DATA_SIZE * 0.4
TEST_EPOCH_SIZE=int(TEST_DATA_SIZE / TEST_BATCH_SIZE)

#定义主函数并执行
def main():
    #train_data = data_init()
    with open('../model_data/data1.18876', 'r') as f:
        rows = f.read().split('\n')
        data1 = [one.split() for one in rows]
        for one in data1:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open('../model_data/data2.18876', 'r') as f:
        rows = f.read().split('\n')
        data2 = [one.split() for one in rows]
        for one in data2:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open('../model_data/target.18876', 'r') as f:
        rows = f.read().split('\n')
        target = [one.split() for one in rows]
        for one in target:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open('../model_data/vocab.100000', 'r') as f:
        global char_set
        char_set = f.read().split('\n')
    target = sum(target, [])

    DATA_SIZE=len(target)
    train_data_size=int(DATA_SIZE*0.6)
    train_data=(data1[0:train_data_size],data2[0:train_data_size],target[0:train_data_size])
    test_data=(data1[train_data_size:DATA_SIZE],data2[train_data_size:DATA_SIZE],target[train_data_size:DATA_SIZE])

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=True, initializer=initializer):
        eval_model = Proofreading_Model(False, TEST_BATCH_SIZE, GRAM)

    saver = tf.train.Saver()
    with tf.Session() as session:
        # 读取模型
        print("loading model...")
        saver.restore(session, "../ckpt/model.ckpt")
        # 测试模型。
        file = open('../test_results.txt', 'w')
        print("In testing:")
        run_epoch(session, eval_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_EPOCH_SIZE, char_set, file)
        file.close()
        saver.save(session, "../ckpt/model.ckpt")
if __name__ == "__main__":
    main()
