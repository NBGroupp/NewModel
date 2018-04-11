# -*- coding: utf-8 -*-
# =====================================================================
import os
import numpy as np
import math
from paras import *

def test(data_tuple):
    test_data0, test_data1, test_data2, test_data3, count = data_tuple
    print("testing...")

    TEST_DATA_SIZE = 1
    # 测试
    for i in range(TEST_DATA_SIZE):

        # 计算特征出现次数和
        sum_bi_left = 0
        sum_bi_right = 0
        sum_tri_left = 0
        sum_tri_right = 0
        sum_tri = 0
        for ele in test_data3[i]:
            bi_left_word = test_data1[i][-1] + ' ' + ele
            if bi_left_word in count:
                sum_bi_left += count[bi_left_word]

            bi_right_word = ele + ' ' + test_data1[i][-1]
            if bi_right_word in count:
                sum_bi_right += count[bi_right_word]

            if (len(test_data1[i]) < 2):
                tri_left_word = "none"
            else:
                tri_left_word = test_data1[i][-2] + ' ' + test_data1[i][-1] + ' ' + ele
            if tri_left_word in count:
                sum_tri_left += count[tri_left_word]

            if (len(test_data2[i]) < 2):
                tri_right_word = "none"
            else:
                tri_right_word = ele + ' ' + test_data2[i][-1] + ' ' + test_data2[i][-2]
            if tri_right_word in count:
                sum_tri_right += count[tri_right_word]

            tri_word = test_data1[i][-1] + ' ' + ele + ' ' + test_data2[i][-1]
            if tri_word in count:
                sum_tri += count[tri_word]


        # 计算概率和分数
        p_bi_left = []
        p_bi_right = []
        p_tri_left = []
        p_tri_right = []
        p_tri = []
        score = []
        original_word_score = 0
        for ele in test_data3[i]:
            ele_score = 0
            bi_left_word = test_data1[i][-1] + ' ' + ele
            if bi_left_word not in count:
                prob = 0
            else:
                prob = count[bi_left_word]/sum_bi_left
            ele_score += 0.1*prob
            p_bi_left.append(prob)

            bi_right_word = ele + ' ' + test_data1[i][-1]
            if bi_right_word not in count:
                prob = 0
            else:
                prob = count[bi_right_word]/sum_bi_right
            ele_score += 0.1 * prob
            p_bi_right.append(prob)

            if (len(test_data1[i]) < 2):
                tri_left_word = "none"
            else:
                tri_left_word = test_data1[i][-2] + ' ' + test_data1[i][-1] + ' ' + ele
            if (tri_left_word not in count) or (count[tri_left_word] == 0):
                prob = 0
            else:
                prob = count[tri_left_word] / sum_tri_left
            ele_score += (0.8/3) * prob
            p_tri_left.append(prob)

            if (len(test_data2[i]) < 2):
                tri_right_word = "none"
            else:
                tri_right_word = ele + ' ' + test_data2[i][-1] + ' ' + test_data2[i][-2]
            if (tri_right_word not in count) or (count[tri_right_word] == 0):
                prob = 0
            else:
                prob = count[tri_right_word] / sum_tri_right
            ele_score += (0.8 / 3) * prob
            p_tri_right.append(prob)

            tri_word = test_data1[i][-1] + ' ' + ele + ' ' + test_data2[i][-1]
            if tri_word not in count:
                prob = 0
            else:
                prob = count[tri_word] / sum_tri
            ele_score += (0.8 / 3) * prob
            p_tri.append(prob)

            score.append(ele_score)
            if(ele == test_data0[i]):
                original_word_score = ele_score


        original_word = int(test_data0[i])
        output_word = original_word

        max_score = -1 # 最大分数
        max_cset_word = -1 # 最大分数对应的字
        for j,ele in enumerate(score):
            if(max_score < ele):
                max_score = ele
                max_cset_word = test_data3[i][j]

        if(original_word_score < max_score):
            if(original_word_score == 0 and max_score > 0):
                output_word = int(max_cset_word)
            elif(original_word_score > 0 and original_word_score < 0.01*max_score):
                output_word = int(max_cset_word)

        return output_word




