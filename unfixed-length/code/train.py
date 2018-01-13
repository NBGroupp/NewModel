# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
import numpy as np
import math
from model import *

NUM_EPOCH = 5 # 迭代次数

TRAIN_BATCH_SIZE = 20 #训练数据batch的大小
#TRAIN_NUM_STEP =35 #训练数据截断长度
# GRAM = 3 #LSTM的时间维度

# DATA_SIZE = 18876
DATA_SIZE = 100
TRAIN_DATA_SIZE = int(DATA_SIZE * 1.0)
TRAIN_EPOCH_SIZE=math.ceil(TRAIN_DATA_SIZE / TRAIN_BATCH_SIZE)

charset=[]
#定义主函数并执行
def main():
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
    # print(target)
    train_data=(data1[0:TRAIN_DATA_SIZE],data2[0:TRAIN_DATA_SIZE],target[0:TRAIN_DATA_SIZE])

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        train_model = Proofreading_Model(True, TRAIN_BATCH_SIZE)

    saver = tf.train.Saver()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    with tf.Session() as session:
        # 训练模型。
        if(os.path.exists("../ckpt/checkpoint")):
            # 读取模型
            print("loading model...")
            saver.restore(session, "./ckpt/model.ckpt")
            file = open('../results/results.txt', 'a')
        else:
            print("new training...")
            tf.global_variables_initializer().run()
            file = open('../results/results.txt', 'w')

        # 记录cost
        # 要使用tensorboard，首先定义summary节点，不定义会出错
        tf.summary.scalar('cost', train_model.cost)
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('../logs/test_logs', session.graph)

        print("In training:")
        for i in range(NUM_EPOCH):
            print("In iteration: %d " % (i + 1))
            file.write("In iteration: %d\n" % (i + 1))
            run_epoch(session, train_model, train_data, train_model.train_op, True,
                      TRAIN_BATCH_SIZE, TRAIN_EPOCH_SIZE, char_set, file, merged_summary_op, summary_writer)

            #保存模型
            print("saving model...")
            saver.save(session, "../ckpt/model.ckpt")
        saver.save(session, "../ckpt/model.ckpt")
        file.close()

if __name__ == "__main__":
    main()
