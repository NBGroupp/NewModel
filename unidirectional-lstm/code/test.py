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
    with open(TARGET_PATH, 'r') as f:
        rows = f.read().strip().split('\n')
        #target = [one.split() for one in rows]
        target = [] 
        for one in rows:
            target.append(int(one))
    with open(VOCAB_PATH, 'r') as f:
        global char_set
        char_set = f.read().split('\n')
    

    test_data=(data1[TRAIN_DATA_SIZE+VALID_DATA_SIZE:DATA_SIZE],target[TRAIN_DATA_SIZE+VALID_DATA_SIZE:DATA_SIZE])
    #test_data=(data1[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],target[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE])
    
    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        test_model = Proofreading_Model(False, TEST_BATCH_SIZE)

    saver = tf.train.Saver()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        # 训练模型。
        # if(os.path.exists("../ckpt/checkpoint")):
        if ckpt and ckpt.model_checkpoint_path:
            # 读取模型
            print("loading model...")
            # saver.restore(session, "../ckpt/model.ckpt")
            saver.restore(session, ckpt.model_checkpoint_path)
            i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
            test_model.global_epoch=i
            test_model.global_step=i*TEST_STEP_SIZE
        else:
            print("model doesn't exist!")
            return
        # 测试模型。
        file = open(TEST_RESULT_PATH, 'w')
        print("In testing with model of epoch %d: " % (i-1))
        run_epoch(session, test_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_BATCH_SIZE, char_set, False, False, False) 
        run_epoch(session, test_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_STEP_SIZE, char_set, file,False,False)
        file.close()

if __name__ == "__main__":
    main()

