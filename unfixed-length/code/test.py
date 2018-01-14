# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
import numpy as np
import math
from model import *
from paras import *

charset=[]
#定义主函数并执行
def main():
    #train_data = data_init()
    with open(DATA1_PATH, 'r') as f:
        rows = f.read().strip().split('\n')
        data1 = [one.split() for one in rows]
        for one in data1:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(DATA2_PATH, 'r') as f:
        rows = f.read().strip().split('\n')
        data2 = [one.split() for one in rows]
        for one in data2:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(TARGET_PATH, 'r') as f:
        rows = f.read().strip().split('\n')
        #target = [one.split() for one in rows]
        target = [] 
        for one in rows:
            target.append(int(one))
    with open(VOCAB_PATH, 'r') as f:
        global char_set
        char_set = f.read().split('\n')
    

    test_data=(data1[TRAIN_DATA_SIZE:DATA_SIZE],data2[TRAIN_DATA_SIZE:DATA_SIZE],target[TRAIN_DATA_SIZE:DATA_SIZE])

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        eval_model = Proofreading_Model(False, TEST_BATCH_SIZE)

    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        # 读取模型
        print("loading model...")
        saver.restore(session, "../ckpt/model.ckpt")
        # 测试模型。
        file = open('../results/test_results.txt', 'w')
        print("In testing:")
        run_epoch(session, eval_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_EPOCH_SIZE, char_set, file,False,False)
        file.close()

if __name__ == "__main__":
    main()

