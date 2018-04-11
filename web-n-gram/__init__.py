#!/usr/bin/python3
#coding: utf-8

import sys
import pickle

from flask import Flask
import numpy as np

print('loading vocab...')
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
    vocab_d = {ch: i for i, ch in enumerate(vocab)}
    UNK_INDEX = 0

print('loading near words dict...')
with open('./nwd.pkl', 'rb') as f:
    nwd = pickle.load(f)

with open('count.dict', 'r') as f:
    dict = f.read()
    count = eval(dict)

app = Flask(__name__)
