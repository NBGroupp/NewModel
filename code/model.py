# -- coding: utf-8 --
#=====================================================================
import tensorflow as tf
import os
import numpy as np
import train

#参数
VOCAB_SIZE = 100000 #词典规模
MAX_TEXT_LENGTH = 50 #最长文本长度

LEARNING_RATE = 0.01 #学习率
LEARNING_RATE_DECAY_FACTOR =  0.9999 #控制学习率下降的参数
KEEP_PROB = 0.65 #节点不Dropout的概率
# MAX_GRAD_NORM = 5 #用于控制梯度膨胀的参数

HIDDEN_SIZE = 100 #词向量维度
PRE_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #上文lstm的隐藏层数目
PRE_CONTEXT_NUM_LAYERS = 1 #上文lstm的深度
FOL_CONTEXT_HIDDEN_SIZE = HIDDEN_SIZE #下文lstm的隐藏层数目
FOL_CONTEXT_NUM_LAYERS= 1 #下文lstm的深度

class Proofreading_Model(object):
    def __init__(self, is_training, batch_size, num_steps):
        """
        :param is_training: is or not training, True/False
        :param batch_size: the size of one batch
        :param num_steps: the length of one lstm
        """
        #定义网络参数
        self.learning_rate = tf.Variable(float(LEARNING_RATE), trainable=False, dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * LEARNING_RATE_DECAY_FACTOR)
        # self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层,其维度是batch_size * num_steps
        self.pre_input = tf.placeholder(tf.int32, [self.batch_size, num_steps])
        self.fol_input = tf.placeholder(tf.int32, [self.batch_size, num_steps])
        # 定义预期输出，它的维度和上面维度相同
        self.targets = tf.placeholder(tf.int32, [self.batch_size,])

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
            self.pre_initial_state = pre_lstm_cell.zero_state(self.batch_size, tf.float32)  # 初始化最初的状态。
            pre_outputs, pre_states = tf.nn.dynamic_rnn(pre_lstm_cell, pre_input, initial_state=self.pre_initial_state,
                                                            dtype=tf.float32)
            #tmp_output = pre_outputs[:, -1, :]    #上一时刻的输出作下一时刻预测的输入
            #pre_outputs, pre_states = pre_lstm_cell(tmp_output, pre_states)
            pre_outputs = pre_outputs[:, -1, :]
            self.pre_final_state = pre_states  #上文LSTM的最终状态

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
            self.fol_initial_state = fol_lstm_cell.zero_state(self.batch_size, tf.float32)  # 初始化最初的状态。
            fol_outputs, fol_states = tf.nn.dynamic_rnn(fol_lstm_cell, fol_input, initial_state=self.fol_initial_state,
                                                        dtype=tf.float32)
            #tmp_output = fol_outputs[:, -1, :]  # 上一时刻的输出作下一时刻预测的输入
            #fol_outputs, fol_states = fol_lstm_cell(tmp_output, fol_states)
            fol_outputs = fol_outputs[:, -1, :]
            self.fol_final_state = fol_states  #下文lstm的最终状态

        # 综合两个lstm的数据，加权平均
        self.output = tf.add(pre_outputs,fol_outputs)/2
        # 全连接层
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        self.logits = tf.matmul(self.output, weight) + bias

        # 简单拼接
        # output = tf.concat([pre_outputs, fol_outputs], 1)
        # 全连接层
        # weight = tf.get_variable("weight", [2*HIDDEN_SIZE, VOCAB_SIZE])
        # bias = tf.get_variable("bias", [VOCAB_SIZE])
        # self.logits = tf.matmul(output, weight) + bias


        ''' 定义交叉熵损失函数和平均损失。
        logits中在vocab_size个结果中选择概率最大的结果与相应的targets结果比较计算loss值
         返回一个 [batch_size] 的1维张量 '''
        #loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #    [self.logits], [self.targets],
        #    [tf.ones([batch_size], dtype=tf.float32)])

        #softmax+交叉熵损失函数
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets)

        self.cost = tf.reduce_mean(loss)
        # 只在训练模型时定义反向传播操作。
        if not is_training: return

        # trainable_variables = tf.trainable_variables()
        # trainable_variables = tf.all_variables()
        # 控制梯度大小，定义优化方法和训练步骤。
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        # self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)


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
    fol_state = session.run(model.fol_initial_state)

    #获取数据
    dataX1, dataX2, dataY = data
    max_cnt = len(dataY)  #  数据长度
    cnt = 0  # 现在取第cnt个输入
    correct_num = 0  #  正确个数

    # 训练一个epoch。
    for step in range(step_size):
        if (cnt + batch_size > max_cnt):  #  如果此时取的数据超过了结尾，就取结尾的batch_size个数据
            cnt = max_cnt - batch_size
        x1 = dataX1[cnt:cnt + batch_size]  #  取前文
        # print(x1)
        x2 = dataX2[cnt:cnt + batch_size]  #  取后文
        y = dataY[cnt:cnt + batch_size]  #  取结果
        # print(y)
        #lstm迭代计算
        cost, pre_state, fol_state, outputs, _, __ = session.run([model.cost, model.pre_final_state, model.fol_final_state,
                                                        model.logits, train_op, learning_rate_decay_op],
                                                       feed_dict={model.pre_input: x1, model.fol_input: x2,
                                                                  model.targets: y,
                                                                  model.pre_initial_state: pre_state,
                                                                  model.fol_initial_state: fol_state
                                                                  })
        total_costs += cost  #  求得总costs
        classes = np.argmax(outputs, axis=1)
        target_index = np.array(y).ravel()
        correct_num = correct_num + sum(classes == target_index)
        # 写入到文件以及输出到屏幕
        #if is_training and (step+1) % 100 == 0:
        if (step+1) % 10 == 0:
            print("After %d steps, cost : %.3f" % (step, total_costs / (step + 1)))
            file.write("After %d steps, cost : %.3f" % (step, total_costs / (step + 1)) + '\n')
            print("outputs: " + ' '.join([char_set[t] for t in classes]))
            print("targets: " + ' '.join([char_set[t] for t in target_index]))
            file.write("outputs: " + ' '.join([char_set[t] for t in classes]) + '\n')
            file.write("targets: " + ' '.join([char_set[t] for t in target_index]) + '\n')


            if(is_training):
                summary_str = session.run(summary_op, feed_dict={model.pre_input: x1,
                                                                        model.fol_input: x2,
                                                                        model.targets: y})
                summary_writer.add_summary(summary_str, step)

        cnt += batch_size
        if (cnt >= max_cnt):
            cnt = 0

    if not is_training:
       acc = correct_num*1.0 / len(dataY) # 求得准确率=正确分类的个数
       print("acc: %.5f" % acc)
       file.write("acc: %.5f" % acc)
       #file.write("acc:" + str(acc))


