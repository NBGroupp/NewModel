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
    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        data1 = [one.split() for one in rows]
        for one in data1:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        data2 = [one.split() for one in rows]
        for one in data2:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        data3 = [one.split() for one in rows]
        for one in data3:
            for index, ele in enumerate(one):
                one[index]=int(ele)
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        #target = [one.split() for one in rows]
        target = [] 
        for one in rows:
            target.append(int(one))
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        global char_set
        char_set = f.read().split('\n')
    
    # print(target)

    train_data=(data1[0:TRAIN_DATA_SIZE],data2[0:TRAIN_DATA_SIZE],data3[0:TRAIN_DATA_SIZE],target[0:TRAIN_DATA_SIZE])
    valid_data=(data1[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],
                data2[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],
                data3[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE + VALID_DATA_SIZE],
                target[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE])
    test_data=(data1[TRAIN_DATA_SIZE+VALID_DATA_SIZE:DATA_SIZE],data2[TRAIN_DATA_SIZE+VALID_DATA_SIZE:DATA_SIZE],
               data3[TRAIN_DATA_SIZE + VALID_DATA_SIZE:DATA_SIZE],target[TRAIN_DATA_SIZE+VALID_DATA_SIZE:DATA_SIZE])
    initializer = tf.random_uniform_initializer(-0.01, 0.01)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        train_model = Proofreading_Model(True, TRAIN_BATCH_SIZE)

    with tf.variable_scope("Proofreading_model", reuse=True, initializer=initializer):
        eval_model = Proofreading_Model(False, VALID_BATCH_SIZE)

    saver = tf.train.Saver()
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    with tf.Session() as session:
        ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
        # 训练模型。
        #if(os.path.exists("../ckpt/checkpoint")):
        i = 0
        if ckpt and ckpt.model_checkpoint_path:
            # 读取模型
            print("loading model...")
            #saver.restore(session, "../ckpt/model.ckpt")
            saver.restore(session, ckpt.model_checkpoint_path)
            i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
            file = open(RESULT_PATH, 'a')
            train_model.global_epoch=eval_model.global_epoch=i
            eval_model.global_step=i*VALID_STEP_SIZE
            train_model.global_step=i*TRAIN_STEP_SIZE
            #run_epoch(session, train_model, train_data, tf.no_op(), False,
                      #TRAIN_BATCH_SIZE, TRAIN_STEP_SIZE, char_set, False, False, False)
        else:
            print("new training...")
            tf.global_variables_initializer().run()
            file = open(RESULT_PATH, 'w')

        # 记录cost
        # 要使用tensorboard，首先定义summary节点，不定义会出错
        merged_summary_op = train_model.merged_summary_op
        summary_writer = tf.summary.FileWriter(COST_PATH, session.graph)

        #PRE_NUM_EPOCH = i
        while i < NUM_EPOCH:
            print("In iteration: %d " % i)
            file.write("In iteration: %d\n" % i)
            
            print("In training:")
            file.write("In training:\n")
            run_epoch(session, train_model, train_data, train_model.train_op, True,
                      TRAIN_BATCH_SIZE, TRAIN_STEP_SIZE, char_set, file, merged_summary_op, summary_writer)
            
            #保存模型
            print("saving model...")
            saver.save(session, CKPT_PATH+MODEL_NAME, global_step = i)
            
            #验证集
            print("In evaluating:")
            file.write("In evaluating:\n")
            run_epoch(session, eval_model, valid_data, tf.no_op(), False,
                      VALID_BATCH_SIZE, VALID_STEP_SIZE, char_set, file, False, False)
            i += 1
            train_model.global_epoch += 1
            
        #saver.save(session, CKPT_PATH+MODEL_NAME, global_step = NUM_EPOCH)
        file.close()
        # 测试模型。
        file = open(TEST_RESULT_PATH, 'w')
        print("In testing with model of epoch %d: " % i-1)
        file.write("In testing with model of epoch %d: \n" % i-1)
        #run_epoch(session, eval_model, test_data, tf.no_op(), False,
                  #TEST_BATCH_SIZE, TEST_STEP_SIZE, char_set, False,False,False)
        run_epoch(session, eval_model, test_data, tf.no_op(), False,
                  TEST_BATCH_SIZE, TEST_STEP_SIZE, char_set, file,False,False)
        file.close()    
        

if __name__ == "__main__":
    main()
