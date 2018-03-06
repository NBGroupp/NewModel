# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
import numpy as np
import time
import random
from paras import *

class Proofreading_Model(object):
    def __init__(self, is_training, batch_size):
        """
        :param is_training: is or not training, True/False
        :param batch_size: the size of one batch
        :param num_steps: the length of one lstm
        """
        #定义网络参数
        self.learning_rate = tf.Variable(float(LEARNING_RATE), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * LEARNING_RATE_DECAY_FACTOR)
        self.global_step = 0
        self.global_epoch = 0
        self.batch_size = batch_size

        # 定义输入层,其维度是batch_size * num_steps
        self.pre_input = tf.placeholder(tf.int32, [batch_size,None])
        self.pre_input_seq_length = tf.placeholder(tf.int32, [batch_size,])

        self.candidate_words_input = tf.placeholder(tf.int32, [batch_size,None])
        self.candidate_in_vocab = tf.placeholder(tf.float32, [batch_size,None])
        # 定义预期输出，它的维度和上面维度相同
        self.targets = tf.placeholder(tf.int32, [batch_size,])
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])  # embedding矩阵
        #self.embedding = embedding
        # pre_context_model
        with tf.variable_scope('Pre') as scope:
            pre_cell = tf.contrib.rnn.BasicLSTMCell(num_units=PRE_CONTEXT_HIDDEN_SIZE, forget_bias=0.0,
                                                state_is_tuple=True)
            if is_training:
                pre_cell = tf.contrib.rnn.DropoutWrapper(pre_cell, output_keep_prob=KEEP_PROB)
            pre_lstm_cell = tf.contrib.rnn.MultiRNNCell([pre_cell] * PRE_CONTEXT_NUM_LAYERS, state_is_tuple=True)

            pre_input = tf.nn.embedding_lookup(embedding, self.pre_input)  # 将原本单词ID转为单词向量。
            if is_training:
                pre_input = tf.nn.dropout(pre_input, KEEP_PROB)
            self.pre_initial_state = pre_lstm_cell.zero_state(batch_size, tf.float32)  # 初始化最初的状态。
            pre_outputs, pre_states = tf.nn.dynamic_rnn(pre_lstm_cell, pre_input,sequence_length=self.pre_input_seq_length,
                                                        initial_state=self.pre_initial_state,dtype=tf.float32)
            #pre_outputs = pre_outputs[:, -1, :]
            pre_outputs = pre_states
            self.pre_final_state = pre_states  #上文LSTM的最终状态


        # 简单拼接
        concat_output = pre_outputs[0][0]
        #print(concat_output)
        #self.concat_output = concat_output
        # 双线性attention
        with tf.variable_scope('bilinear'):  # Bilinear Layer (Attention Step)
            candidate_words_input_vector = tf.nn.embedding_lookup(embedding, self.candidate_words_input)
            bilinear_weight = tf.get_variable("bilinear_weight", [HIDDEN_SIZE, HIDDEN_SIZE])  # embedding矩阵
            '''计算候选词与上下文的匹配度'''
            M = candidate_words_input_vector * tf.expand_dims(tf.matmul(concat_output, bilinear_weight), axis=1)  # M = [batch_size,candi_num,hidden_size]
            #self.M = M
            # attention概率(匹配度)
            alpha = tf.nn.softmax(tf.reduce_sum(M, axis=2))  # [batch_size,candi_num]
            #self.alpha = alpha
            # sum attention 模型
            bilinear_output = tf.reduce_sum(candidate_words_input_vector * tf.expand_dims(alpha, axis=2), axis=1)  # [batch, h]

        # 全连接层
        # weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        # bias = tf.get_variable("bias", [VOCAB_SIZE])
        # self.logits = tf.matmul(output, weight) + bias

        # sotfmax层
        #self.bilinear_output = bilinear_output
        softmax_prob = tf.layers.dense(bilinear_output, units=VOCAB_SIZE, activation=tf.nn.softmax,
                                         kernel_initializer=tf.random_uniform_initializer(-0.01,0.01))  # [batch_size, vocab_size#]

        #self.softmax_prob = softmax_prob
        # 非候选词概率置0
        tmp_prob = softmax_prob * self.candidate_in_vocab
        #self.tmp_prob = tmp_prob
        # 重算概率
        self.logits = tmp_prob / tf.expand_dims(tf.reduce_sum(tmp_prob, axis=1),axis=1)
        self.logits = tf.clip_by_value(self.logits, 1e-7, 1.0 - 1e-7)

        ''' 定义交叉熵损失函数和平均损失。
        logits中在vocab_size个结果中选择概率最大的结果与相应的targets结果比较计算loss值
         返回一个 [batch_size] 的1维张量 '''
        # loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #    [self.logits], [self.targets],
        #    [tf.ones([batch_size], dtype=tf.float32)])

        # softmax + 交叉熵损失函数
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)

        # 求交叉熵
        one_hot_labels = tf.one_hot(self.targets,VOCAB_SIZE)
        loss = -tf.reduce_sum(one_hot_labels * tf.log(self.logits),reduction_indices = 1)
        # 记录cost
        with tf.variable_scope('cost'):
            self.cost = tf.reduce_mean(loss)
            self.ave_cost = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.ave_cost_op = self.ave_cost.assign(tf.divide(
                tf.add(tf.multiply(self.ave_cost, self.global_step), self.cost), self.global_step+1))
            #global_step从0开始
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('ave_cost', self.ave_cost)
        # 只在训练模型时定义反向传播操作。

        # 记录accuracy
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(self.targets, tf.cast(tf.argmax(self.logits, -1), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.ave_accuracy = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.ave_accuracy_op =  self.ave_accuracy.assign(tf.divide(
                tf.add(tf.multiply(self.ave_accuracy, self.global_step),self.accuracy),self.global_step+1))
            # global_step从0开始
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('ave_accuracy', self.ave_accuracy)
            # 只在训练模型时定义反向传播操作。
        # 只在训练模型时定义反向传播操作。
        if not is_training: return

        # trainable_variables = tf.trainable_variables()
        # trainable_variables = tf.all_variables()
        # 控制梯度大小，定义优化方法和训练步骤。
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        #self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.cost)

        self.merged_summary_op = tf.summary.merge_all() # 收集节点
        #self.merged_summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES,scope))
        
# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的cost值
def run_epoch(session, model, data, train_op, is_training, batch_size, step_size, char_set, file,
              summary_op, summary_writer):
    """
    :param session: tf.Session() to compute
    :param model: the proof model already defined
    :param data: data for running
    :param train_op: train operation
    :param is_training: is or not training
    :param batch_size: the size of one batch
    :param step_size: the number of step to run the model
    :param char_set: the dictionary
    :param file: file to write
    :param summary_op: the operation to merge the parameters
    :param summary_writer: output graph writer
    :return: none
    """
    #总costs
    total_costs = 0.0

    #初始化数据
    pre_state = session.run(model.pre_initial_state)

    #获取数据
    dataX1, dataX3,dataY = data
    max_cnt = len(dataY)  #  数据长度
    if is_training:
        cnt = random.randint(0,max_cnt-batch_size+1)  # 现在取第cnt个输入
    else:
        cnt = 0 
    correct_num = 0  #  正确个数

    # 训练一个epoch。
    start = time.clock()
    for step in range(step_size):
        if (cnt + batch_size > max_cnt):  #  如果此时取的数据超过了结尾，就取结尾的batch_size个数据
            cnt = max_cnt - batch_size
        x1 = dataX1[cnt:cnt + batch_size]  #  取前文
        x1,x1_seqlen = Pad_Zero(x1)# 补0
        # print(x1)

        x3 = dataX3[cnt:cnt + batch_size]
        x4 = is_candidate(x3)
        x3, _ = Pad_Zero(x3)

        y = dataY[cnt:cnt + batch_size]  #  取结果
        # print(y)
        #lstm迭代计算

        cost, pre_state, fol_state, outputs, _, _, ave_cost_op, ave_accuracy_op\
        = session.run([model.cost, model.pre_final_state,
                                                        model.logits, train_op, model.learning_rate_decay_op,
                                                        model.ave_cost_op, model.ave_accuracy_op],
                                                       feed_dict={model.pre_input: x1,
                                                                  model.candidate_words_input: x3,
                                                                  model.candidate_in_vocab: x4,
                                                                  model.pre_input_seq_length:x1_seqlen,
                                                                  model.targets: y,
                                                                  model.pre_initial_state: pre_state
                                                                  })
        if (is_training):
            model.global_step+=1
        cnt += batch_size
        if (cnt >= max_cnt):
            cnt = 0        
        if not file:
            continue     
        total_costs += cost  #  求得总costs
        classes = np.argmax(outputs, axis=1)
        target_index = np.array(y).ravel()
        correct_num = correct_num + sum(classes == target_index)
        
        # 写入到文件以及输出到屏幕
        if (((step+1) % stepinter == 0) or ( step == 0 )) and file:
            '''
            # 输出softmax概率值
            print("logits sum: ",np.sum(outputs,1))
            print("target prob: ")
            for i in range(batch_size):
                print(outputs[i][y[i]],end=' ') 
            print("\noutput prob: ")
            for i in range(batch_size):
                print(outputs[i][classes[i]],end=' ')  
            print("\nclasses: ",classes)
            '''
            end = time.clock()
            print("%.1f setp/s" % (stepinter/(end-start)))
            start = time.clock()
            print("After %d steps, cost : %.3f" % (step, total_costs / (step + 1)))
            file.write("After %d steps, cost : %.3f" % (step, total_costs / (step + 1)) + '\n')
            print("outputs: " + ' '.join([char_set[t] for t in classes]))
            print("targets: " + ' '.join([char_set[t] for t in target_index]))
            file.write("outputs: " + ' '.join([char_set[t] for t in classes]) + '\n')
            file.write("targets: " + ' '.join([char_set[t] for t in target_index]) + '\n')

    if file:
        print("After this epoch, cost : %.3f" % (total_costs / (step_size)))
        file.write("After this epoch, cost : %.3f" % (total_costs / (step_size)) + '\n')
    
    #收集并将cost加入记录
    if(is_training):
        # print ('ave_cost = %.5f' % (total_costs / (step_size + 1)))
        summary_str = session.run(summary_op, feed_dict={model.pre_input: x1,
                                                         model.candidate_words_input: x3,
                                                         model.candidate_in_vocab: x4,
                                                                  model.pre_input_seq_length:x1_seqlen,
                                                                  model.targets: y,
                                                                  model.pre_initial_state: pre_state
                                                                  })
        summary_writer.add_summary(summary_str, model.global_epoch)
    if not is_training and file:
       acc = correct_num*1.0 / len(dataY) # 求得准确率=正确分类的个数
       print("acc: %.5f\n" % acc)
       file.write("acc: %.5f\n" % acc)

def Pad_Zero(x):
    x_seqlen=[]
    row_len = len(x)
    max_len=0
    for i in range(row_len):
        col_len = len(x[i])
        x_seqlen.append(col_len)
        max_len = max(max_len, col_len)

    for i in range(row_len):
        col_len=x_seqlen[i]
        for j in range(col_len,max_len):
            x[i].append(0)

    # print(x)
    # print(x_seqlen)
    return x,x_seqlen

def is_candidate(x):
    is_candi = np.zeros([len(x),VOCAB_SIZE],dtype=np.float32)
    for i,index in enumerate(x):
        is_candi[i,index] = 1.0
    #for i in range(len(x)):
        #is_candi[i][0] = 0.0
    return is_candi
