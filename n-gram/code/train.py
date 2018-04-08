# -*- coding: utf-8 -*-
#=====================================================================
import os
import numpy as np
import math
from paras import *

def main():
    with open(DATA1_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        data1 = [one.split() for one in rows]
        # for one in data1:
        #     for index, ele in enumerate(one):
        #         one[index]=int(ele)
    with open(DATA2_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        data2 = [one.split() for one in rows]
        # for one in data2:
        #     for index, ele in enumerate(one):
        #         one[index]=int(ele)
    with open(DATA3_PATH, 'r', encoding='utf-8') as f:
        rows = f.read().strip().split('\n')
        data3 = [one.split() for one in rows]
        # for one in data3:
        #     for index, ele in enumerate(one):
        #         one[index]=int(ele)
    with open(TARGET_PATH, 'r', encoding='utf-8') as f:
        target = f.read().strip().split('\n')
        # rows = f.read().strip().split('\n')
        # target = []
        # for one in rows:
        #     target.append(int(one))
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        global char_set
        char_set = f.read().split('\n')

    train_data = (data1[0:TRAIN_DATA_SIZE], data2[0:TRAIN_DATA_SIZE], data3[0:TRAIN_DATA_SIZE], target[0:TRAIN_DATA_SIZE])

    train_data1, train_data2, train_data3, train_target = train_data

    # 训练
    print("training...")
    count = { "none":0 }
    for i in range(TRAIN_DATA_SIZE):
        if(((i+1) % 10000 == 0) or (i == 0)):
            print("training %d row" % (i+1))
        bi_word = train_data1[i][-1] + ' ' + train_target[i]
        tri__word = train_data1[i][-1] + ' ' + train_target[i] + ' ' + train_data2[i][-1]
        if bi_word not in count:
            count[bi_word] = 0
        if tri__word not in count:
            count[tri__word] = 0
        count[bi_word] += 1
        count[tri__word] += 1
    
    print("saving count dict...")
    with open(COUNT_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write(str(count))

if __name__ == "__main__":
    main()
