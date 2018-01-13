# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
import numpy as np
import math
from model import *

TEST_BATCH_SIZE = 20 #测试数据batch的大小
#TEST_NUM_STEP = 1 #测试数据截断长度
GRAM = 3 #LSTM的时间维度

DATA_SIZE = 18876
TRAIN_DATA_SIZE = int(DATA_SIZE * 0.7)
TEST_DATA_SIZE = int(DATA_SIZE-TRAIN_DATA_SIZE)
TEST_EPOCH_SIZE=math.ceil(TEST_DATA_SIZE / TEST_BATCH_SIZE)

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

    test_data=(data1[TRAIN_DATA_SIZE:DATA_SIZE],data2[TRAIN_DATA_SIZE:DATA_SIZE],target[TRAIN_DATA_SIZE:DATA_SIZE])

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        eval_model = Proofreading_Model(False, TEST_BATCH_SIZE, GRAM)

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        # 读取模型
        print("loading model...")
        saver.restore(session, "../ckpt/model.ckpt")
        # 测试模型。
        file = open('../test_results.txt', 'w')
        print("In testing:")
        run_epoch(session, eval_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_EPOCH_SIZE, char_set, file,False,False)
        file.close()

if __name__ == "__main__":
    main()

