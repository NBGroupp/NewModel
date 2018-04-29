# -- coding: utf-8 --
# =====================================================================
import tensorflow as tf
import os
import numpy as np
import time
import random
from modules import *
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
        self.candidate_len = tf.reduce_sum(self.is_candidate, reduction_indices=1)

        self.one_hot_labels = tf.placeholder(tf.float32, [batch_size, None])

        # 定义预期输出，它的维度和上面维度相同
        self.targets = tf.placeholder(tf.int32, [batch_size, ])
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])  # embedding矩阵
        candidate_words_input_vector = tf.nn.embedding_lookup(embedding, self.candidate_words_input) #(batchsize * candidate_len * hidden_size)
        # pre_context_model
        with tf.variable_scope('Pre') as scope:
            pre_output = tf.nn.embedding_lookup(embedding, self.pre_input)  # 将原本单词ID转为单词向量。
           
            ## Positional Encoding
            pre_output += positional_encoding(self.pre_input,
                              num_units=HIDDEN_SIZE, 
                              zero_pad=False, 
                              scale=False,
                              scope="Pre_pe")
            ## Dropout
            if is_training:
                pre_output = tf.nn.dropout(pre_output, KEEP_PROB)
                
            ## self-multihead attention
            pre_output = multihead_attention(queries=pre_outputs, 
                                                    keys=pre_outputs,
                                                    queries_length=self.pre_input_seq_length,
                                                    keys_length=self.pre_input_seq_length,
                                                    num_units=HIDDEN_SIZE, 
                                                    num_heads=NUM_HEADS, 
                                                    dropout_rate=DROP_RATE,
                                                    is_training=is_training,
                                                    causality=False,
                                                    scope="Pre")
            
            ## multihead attention
            pre_attention = multihead_attention(queries=candidate_words_input_vector, 
                                                    keys=pre_outputs,
                                                    queries_length=self.candidate_len,
                                                    keys_length=self.pre_input_seq_length,
                                                    num_units=HIDDEN_SIZE, 
                                                    num_heads=NUM_HEADS, 
                                                    dropout_rate=DROP_RATE,
                                                    is_training=is_training,
                                                    causality=False,
                                                    scope="Pre")
            '''
            self.pre_initial_state = pre_lstm_cell.zero_state(batch_size, tf.float32)  # 初始化最初的状态。
            pre_outputs, pre_states = tf.nn.dynamic_rnn(pre_lstm_cell, pre_input,
                                                        sequence_length=self.pre_input_seq_length,
                                                        initial_state=self.pre_initial_state, dtype=tf.float32)
            # pre_outputs = pre_outputs[:, -1, :]
            pre_outputs = pre_states
            self.pre_final_state = pre_states  # 上文LSTM的最终状态
            '''

        # fol_context_model
        with tf.variable_scope('Fol') as scope:
            fol_output = tf.nn.embedding_lookup(embedding, self.fol_input)  # 将原本单词ID转为单词向量。
            
            ## Positional Encoding
            fol_output += positional_encoding(self.fol_input,
                  num_units=FOL_CONTEXT_HIDDEN_SIZE, 
                  zero_pad=False, 
                  scale=False,
                  scope="Fol_pe")
             
                  
            ## Dropout
            if is_training:
                fol_output = tf.nn.dropout(fol_output, KEEP_PROB)
                
            ## self-multihead attention
            fol_output = multihead_attention(queries=fol_output, 
                                                    keys=fol_output,
                                                    queries_length=self.fol_output_seq_length,
                                                    keys_length=self.fol_output_seq_length,
                                                    num_units=HIDDEN_SIZE, 
                                                    num_heads=NUM_HEADS, 
                                                    dropout_rate=DROP_RATE,
                                                    is_training=is_training,
                                                    causality=False,
                                                    scope="Pre")
                                                    
            ## multihead attention
            fol_attention = multihead_attention(queries=candidate_words_input_vector, 
                                                    keys=fol_outputs,
                                                    queries_length=self.candidate_len,
                                                    keys_length=self.fol_input_seq_length,
                                                    num_units=HIDDEN_SIZE, 
                                                    num_heads=NUM_HEADS, 
                                                    dropout_rate=DROP_RATE,
                                                    is_training=is_training,
                                                    causality=False,
                                                    scope="Pre")                                        
            '''
            self.fol_initial_state = fol_lstm_cell.zero_state(batch_size, tf.float32)  # 初始化最初的状态。
            fol_outputs, fol_states = tf.nn.dynamic_rnn(fol_lstm_cell, fol_input,
                                                        sequence_length=self.fol_input_seq_length,
                                                        initial_state=self.fol_initial_state,
                                                        dtype=tf.float32)
            # fol_outputs = fol_outputs[:, -1, :]
            fol_outputs = fol_states
            self.fol_final_state = fol_states  # 下文lstm的最终状态
            '''
        
        
        
        # 简单拼接
        concat_output = tf.concat([pre_attention, fol_attention], axis=-1)
        feed_output = tf.reduce_sum(feedforward(concat_output, num_units=[1], scope="feed_forward"), reduction_indices=-1)
        alpha = tf.nn.softmax(feed_output)  # [batch_size,candi_num]
        # 非候选词概率置0
        tmp_prob = alpha * self.is_candidate

        # 重算概率
        self.logits = tmp_prob / tf.expand_dims(tf.reduce_sum(tmp_prob, axis=1), axis=1)
        self.logits = tf.clip_by_value(self.logits, 1e-7, 1.0 - 1e-7)
        
        # 求交叉熵
        self.y_smoothed = label_smoothing(self.one_hot_labels)
        loss = -tf.reduce_sum(self.y_smoothed * tf.log(self.logits), reduction_indices=1)

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
    dataX0, dataX1, dataX2, dataX3, dataY = data
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
        x0 = dataX0[cnt:cnt + batch_size]   

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
        output_classes = np.argmax(outputs, axis=1)

        original_classes = []
        for i in range(len(x0)):
            for j,ele in enumerate(x3[i]):
                if(x0[i] == ele):
                     original_classes.append(j)

        classes = [x3[i][j] for i,j in enumerate(output_classes)]
        target_index = np.array(y).ravel()
        
        correct_num = correct_num + sum(classes == target_index)

        # 统计评价参数
        statistics_evaluation(classes, target_index, x0, outputs,output_classes, original_classes)

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
        print_evaluation(file)


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


def statistics_evaluation(classes,target_index,x0, prob_logits,output_classes, original_classes):
    global TP, FP, TN, FN, P, N, TPW, TPR
    for i,output_word in enumerate(classes):
        original_word = x0[i]
        target_word = target_index[i]
        
        if(prob_logits[i][original_classes[i]] > PROOFREAD_BIAS * prob_logits[i][output_classes[i]]):
            output_word = original_word
        '''
        if(prob_logits[i][output_classes[i]] < PROOFREAD_BIAS):
              output_word = original_word
        '''
        # print("output:%.3lf, original:%.3lf" % (prob_logits[i][output_classes[i]],prob_logits[i][original_classes[i]]))
 
        if (output_word != original_word):  # 修改的文本
            if (original_word != target_word):#错改对或错改错
                TP = TP + 1
                if (output_word == target_word): #错改对
                    TPR = TPR +1
                if (output_word != target_word): #错改错
                    TPW = TPW +1
            elif (original_word == target_word) and (output_word != target_word): #对改错
                FP = FP + 1
        else:  # 不修改的文本
            if (original_word == target_word):
                TN = TN + 1
            else:
                FN = FN + 1


def print_evaluation(file):
    global TP, FP, TN, FN, P, N, TPW, TPR
    P = TP + FN
    N = TN + FP
    print("P : %d\t N : %d" % (P,N))
    file.write("P : %d\t N : %d\n" % (P,N))
    print("TP : %d\t FP : %d" % (TP, FP))
    file.write("TP : %d\t FP : %d\n" % (TP, FP))
    print("TN : %d\t FN : %d" % (TN, FN))
    file.write("TN : %d\t FN : %d\n" % (TN, FN))
    print("TPR : %d\t TPW : %d" % (TPR, TPW))
    file.write("TPR : %d\t TPW : %d" % (TPR, TPW))

    Accuracy = (TP+TN)/(P+N)
    Error_Rate = 1-Accuracy
    Recall = TP/P
    Precision = TP/(TP+FP)
    F1_Score = 2*Precision*Recall/(Precision+Recall)
    Correction_Rate = TPR / TP
    Specificity = TN / N
    Delta = (P-(FP+FN+TPW)) / P 
    print("Accuracy : %.5f " % Accuracy)
    file.write("Accuracy : %.5f \n" % Accuracy)
    print("Error_Rate : %.5f " % Error_Rate)
    file.write("Error_Rate : %.5f \n" % Error_Rate)
    print("Recall : %.5f " % Recall)
    file.write("Recall : %.5f \n" % Recall)
    print("Precision : %.5f " % Precision)
    file.write("Precision : %.5f \n" % Precision)
    print("F1_Score : %.5f " % F1_Score)
    file.write("F1_Score : %.5f \n" % F1_Score)
    print("Correction_Rate : %.5f " % Correction_Rate)
    file.write("Correction_Rate : %.5f \n" % Correction_Rate)
    print("Specificity : %.5f " % Specificity)
    file.write("Specificity : %.5f \n" % Specificity)
    print("Delta : %.5f " % Delta)
    file.write("Delta : %.5f \n" % Delta)