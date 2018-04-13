# -- coding: utf-8 --
# =====================================================================
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
        # 定义网络参数
        self.learning_rate = tf.Variable(float(LEARNING_RATE), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * LEARNING_RATE_DECAY_FACTOR)
        self.global_step = 0
        self.global_epoch = 0
        self.batch_size = batch_size

        # 定义输入层,其维度是batch_size * num_steps
        self.pre_input = tf.placeholder(tf.int32, [batch_size, None])
        self.pre_input_seq_length = tf.placeholder(tf.int32, [batch_size, ])
        self.fol_input = tf.placeholder(tf.int32, [batch_size, None])
        self.fol_input_seq_length = tf.placeholder(tf.int32, [batch_size, ])

        self.candidate_words_input = tf.placeholder(tf.int32, [batch_size, None])
        self.is_candidate = tf.placeholder(tf.float32, [batch_size, None])

        self.one_hot_labels = tf.placeholder(tf.float32, [batch_size, None])

        # 定义预期输出，它的维度和上面维度相同
        self.targets = tf.placeholder(tf.int32, [batch_size, ])
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])  # embedding矩阵
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
            pre_outputs, pre_states = tf.nn.dynamic_rnn(pre_lstm_cell, pre_input,
                                                        sequence_length=self.pre_input_seq_length,
                                                        initial_state=self.pre_initial_state, dtype=tf.float32)
            # pre_outputs = pre_outputs[:, -1, :]
            pre_outputs = pre_states
            self.pre_final_state = pre_states  # 上文LSTM的最终状态

        # fol_context_model
        with tf.variable_scope('Fol') as scope:
            fol_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FOL_CONTEXT_HIDDEN_SIZE, forget_bias=0.0,
                                                    state_is_tuple=True)
            if is_training:
                fol_cell = tf.contrib.rnn.DropoutWrapper(fol_cell, output_keep_prob=KEEP_PROB)
            fol_lstm_cell = tf.contrib.rnn.MultiRNNCell([fol_cell] * FOL_CONTEXT_NUM_LAYERS, state_is_tuple=True)

            fol_input = tf.nn.embedding_lookup(embedding, self.fol_input)  # 将原本单词ID转为单词向量。
            if is_training:
                fol_input = tf.nn.dropout(fol_input, KEEP_PROB)
            self.fol_initial_state = fol_lstm_cell.zero_state(batch_size, tf.float32)  # 初始化最初的状态。
            fol_outputs, fol_states = tf.nn.dynamic_rnn(fol_lstm_cell, fol_input,
                                                        sequence_length=self.fol_input_seq_length,
                                                        initial_state=self.fol_initial_state,
                                                        dtype=tf.float32)
            # fol_outputs = fol_outputs[:, -1, :]
            fol_outputs = fol_states
            self.fol_final_state = fol_states  # 下文lstm的最终状态

        # 简单拼接
        concat_output = tf.concat([pre_outputs[0][-1], fol_outputs[0][-1]], axis=-1)
        # 双线性attention
        with tf.variable_scope('bilinear'):  # Bilinear Layer (Attention Step)
            candidate_words_input_vector = tf.nn.embedding_lookup(embedding, self.candidate_words_input)
            bilinear_weight = tf.get_variable("bilinear_weight", [2 * HIDDEN_SIZE, HIDDEN_SIZE])
            '''计算候选词与上下文的匹配度'''
            M = candidate_words_input_vector * tf.expand_dims(tf.matmul(concat_output, bilinear_weight),
                                                              axis=1)  # M = [batch_size,candi_num,hidden_size]
            # attention概率(匹配度)
            alpha = tf.nn.softmax(tf.reduce_sum(M, axis=2))  # [batch_size,candi_num]

        # 非候选词概率置0
        tmp_prob = alpha * self.is_candidate

        # 重算概率
        self.logits = tmp_prob / tf.expand_dims(tf.reduce_sum(tmp_prob, axis=1), axis=1)
        self.logits = tf.clip_by_value(self.logits, 1e-7, 1.0 - 1e-7)

        # 求交叉熵
        loss = -tf.reduce_sum(self.one_hot_labels * tf.log(self.logits), reduction_indices=1)

        # 记录cost
        with tf.variable_scope('cost'):
            self.cost = tf.reduce_mean(loss)
            self.ave_cost = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.ave_cost_op = self.ave_cost.assign(tf.divide(
                tf.add(tf.multiply(self.ave_cost, self.global_step), self.cost), self.global_step + 1))
            # global_step从0开始
            tf.summary.scalar('cost', self.cost)
            tf.summary.scalar('ave_cost', self.ave_cost)
        # 只在训练模型时定义反向传播操作。

        # 记录accuracy
        with tf.variable_scope('accuracy'):
            correct_prediction = tf.equal(self.targets, tf.cast(tf.argmax(self.logits, -1), tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.ave_accuracy = tf.Variable(0.0, trainable=False, dtype=tf.float32)
            self.ave_accuracy_op = self.ave_accuracy.assign(tf.divide(
                tf.add(tf.multiply(self.ave_accuracy, self.global_step), self.accuracy), self.global_step + 1))
            # global_step从0开始
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('ave_accuracy', self.ave_accuracy)
            # 只在训练模型时定义反向传播操作。
        # 只在训练模型时定义反向传播操作。
        if not is_training: return

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        # self.train_op = optimizer.minimize(self.cost)

        self.merged_summary_op = tf.summary.merge_all()  # 收集节点


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
    # 总costs
    total_costs = 0.0
    # 获取数据
    dataX1, dataX2, dataX3, dataY = data
    max_cnt = len(dataY)  # 数据长度
    if is_training:
        cnt = random.randint(0, max_cnt - batch_size + 1)  # 现在取第cnt个输入
    else:
        cnt = 0
    correct_num = 0  # 正确个数

    # 训练一个epoch。
    start = time.clock()
    for step in range(step_size):
        if (cnt + batch_size > max_cnt):  # 如果此时取的数据超过了结尾，就取结尾的batch_size个数据
            cnt = max_cnt - batch_size
        x1 = dataX1[cnt:cnt + batch_size]  # 取前文
        x1, x1_seqlen = Pad_Zero(x1)  # 补0

        x2 = dataX2[cnt:cnt + batch_size]  # 取后文
        x2, x2_seqlen = Pad_Zero(x2)  # 补0

        x3 = dataX3[cnt:cnt + batch_size]
        x3, _ = Pad_Zero(x3)

        y = dataY[cnt:cnt + batch_size]  # 取结果

        x4, one_hot = is_candidate(x3, y)

        cost, outputs, _, _, ave_cost_op, ave_accuracy_op \
            = session.run([model.cost, model.logits, train_op, model.learning_rate_decay_op,
                           model.ave_cost_op, model.ave_accuracy_op],
                          feed_dict={model.pre_input: x1, model.fol_input: x2,
                                     model.candidate_words_input: x3,
                                     model.is_candidate: x4,
                                     model.pre_input_seq_length: x1_seqlen,
                                     model.fol_input_seq_length: x2_seqlen,
                                     model.targets: y,
                                     model.one_hot_labels: one_hot
                                     })
        if (is_training):
            model.global_step += 1
            cnt = random.randint(0, max_cnt - batch_size + 1)

        else:
            cnt += batch_size
        if (cnt >= max_cnt):
            cnt = 0
        if not file:
            continue
        total_costs += cost  # 求得总costs
        candidate_classes = np.argmax(outputs, axis=1)

        classes = [x3[i][j] for i,j in enumerate(candidate_classes)]
        target_index = np.array(y).ravel()
        correct_num = correct_num + sum(classes == target_index)

        # 写入到文件以及输出到屏幕
        if (((step + 1) % STEP_PRINT == 0) or (step == 0)) and file:
            end = time.clock()
            print("%.1f setp/s" % (STEP_PRINT / (end - start)))
            start = time.clock()
            print("After %d steps, cost : %.3f" % (step+1, total_costs / (step + 1)))
            file.write("After %d steps, cost : %.3f" % (step+1, total_costs / (step + 1)) + '\n')
            print("outputs: " + ' '.join([char_set[t] for t in classes]))
            print("targets: " + ' '.join([char_set[t] for t in target_index]))
            file.write("outputs: " + ' '.join([char_set[t] for t in classes]) + '\n')
            file.write("targets: " + ' '.join([char_set[t] for t in target_index]) + '\n')

    if file:
        print("After this epoch, cost : %.3f" % (total_costs / (step_size)))
        file.write("After this epoch, cost : %.3f" % (total_costs / (step_size)) + '\n')

    # 收集并将cost加入记录
    if (is_training):
        summary_str = session.run(summary_op, feed_dict={model.pre_input: x1, model.fol_input: x2,
                                                         model.candidate_words_input: x3,
                                                         model.is_candidate: x4,
                                                         model.pre_input_seq_length: x1_seqlen,
                                                         model.fol_input_seq_length: x2_seqlen,
                                                         model.targets: y,
                                                         model.one_hot_labels: one_hot
                                                         })
        summary_writer.add_summary(summary_str, model.global_epoch)
    if not is_training and file:
        acc = correct_num * 1.0 / len(dataY)  # 求得准确率=正确分类的个数
        print("acc: %.5f\n" % acc)
        file.write("acc: %.5f\n" % acc)


def Pad_Zero(x):
    x_seqlen = []
    row_len = len(x)
    max_len = 0
    for i in range(row_len):
        col_len = len(x[i])
        x_seqlen.append(col_len)
        max_len = max(max_len, col_len)

    for i in range(row_len):
        col_len = x_seqlen[i]
        for j in range(col_len, max_len):
            x[i].append(0)
    return x, x_seqlen


def is_candidate(x, target):
    x_len = len(x)
    y_len = len(x[0])
    is_candi = np.zeros([x_len, y_len], dtype=np.float32)
    one_hot = np.zeros([x_len, y_len], dtype=np.float32)
    for i in range(x_len):
        for j in range(y_len):
            if(x[i][j] > 0):
                is_candi[i][j] = 1.0
            if(x[i][j] == target[i]):
                one_hot[i][j] = 1.0
    return is_candi, one_hot
