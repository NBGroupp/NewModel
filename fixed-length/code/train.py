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

    #print(data1)

    train_data=(data1[0:TRAIN_DATA_SIZE],data2[0:TRAIN_DATA_SIZE],target[0:TRAIN_DATA_SIZE])
    valid_data=(data1[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],
                data2[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE],target[TRAIN_DATA_SIZE:TRAIN_DATA_SIZE+VALID_DATA_SIZE])

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
        train_model = Proofreading_Model(True, TRAIN_BATCH_SIZE, GRAM)

    with tf.variable_scope("Proofreading_model", reuse=True, initializer=initializer):
        eval_model = Proofreading_Model(False, VALID_BATCH_SIZE, GRAM)

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
            i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            file = open(RESULT_PATH, 'a')
        else:
            print("new training...")
            tf.global_variables_initializer().run()
            file = open(RESULT_PATH, 'w')

        # 记录cost
        # 要使用tensorboard，首先定义summary节点，不定义会出错
        merged_summary_op = train_model.merged_summary_op
        summary_writer = tf.summary.FileWriter(COST_PATH, session.graph)

        #print("In training:")
        #for i in range(NUM_EPOCH):
        while i < NUM_EPOCH:
            print("In training:")
            print("In iteration: %d " % i)
            file.write("In iteration: %d\n" % i)
            run_epoch(session, train_model, train_data, train_model.train_op, True,
                      TRAIN_BATCH_SIZE, TRAIN_STEP_SIZE, char_set, file, merged_summary_op, summary_writer)


            #验证集
            print("In evaluating:")
            run_epoch(session, eval_model, valid_data, tf.no_op(), False,
                      VALID_BATCH_SIZE, VALID_STEP_SIZE, char_set, file, False, False)

            i += 1
            train_model.global_epoch += 1
            #保存模型
            print("saving model...")
            saver.save(session, CKPT_PATH+MODEL_NAME, global_step=i)
        saver.save(session, CKPT_PATH+MODEL_NAME, global_step=NUM_EPOCH)
        file.close()

if __name__ == "__main__":
    main()

