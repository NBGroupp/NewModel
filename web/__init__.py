#!/usr/bin/python3
#coding: utf-8

import sys
import pickle

from flask import Flask
import tensorflow as tf
import numpy as np
from model import *


model = None
session = None
TEST_BATCH_SIZE = 1
initializer = tf.random_uniform_initializer(-0.01, 0.01)
with tf.variable_scope("Proofreading_model", reuse=None, initializer=initializer):
    model = Proofreading_Model(False, TEST_BATCH_SIZE)
saver = tf.train.Saver()
session = tf.Session()
ckpt = tf.train.get_checkpoint_state(CKPT_PATH)
# 训练模型。
if ckpt and ckpt.model_checkpoint_path:
    # 读取模型
    print("loading model...")
    saver.restore(session, ckpt.model_checkpoint_path)
    i = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])+1
    model.global_epoch=i
    model.global_step=i*TEST_STEP_SIZE
else:
    print("model doesn't exist!")
    sys.exit()

print('loading vocab...')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    vocab_d = {ch: i for i, ch in enumerate(vocab)}
    UNK_INDEX = 0

print('loading near words dict...')
with open('./nwd.pkl', 'rb') as f:
    nwd = pickle.load(f)


app = Flask(__name__)
