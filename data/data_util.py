# coding: utf-8

import os
import re
import sys
import gzip
import json
import random
import pickle

import jieba


puncs = "＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"

_PUNC_RE = re.compile('[' + puncs + ']')
_DIGIT_RE = re.compile('[\d１２３４５６７８９０]+\.?[\d１２３４５６７８９０]*%?')
_CHAR_RE = re.compile("[A-Za-zＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ]+")
_OTHERS_RE = re.compile('[^\.a0\u4e00-\u9fff]')

UNK = 'UNK'
UNK_INDEX = 0
PAD = 'PAD'
PAD_INDEX = 1
pre_vocab = [UNK, PAD]


def _is_chinese(c):
    return '\u4e00' <= c <= '\u9fff'

def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


def clean_corpus(data_path, replace=False, strict=False):
    """ Clean corpus data to delete empty lines
        or line has too few Chinese characters(<=3).
        If the corpus is unclean and argument 'replace' is true,
        replace origin corpus_data with new data.
        Args:
            data path: origin corpus file path
            replace: whether replace origin corpus_data with clean data
        Returns:
            new_corpus_data: clean corpus data in list format.
    """
    print('Checking origin data...')
    f = gzip.GzipFile(data_path, mode='r')
    corpus_data = f.read().decode().split('\n')
    f.close()
    new_corpus_data = set()
    drop = 0
    others_re = re.compile('[^' + puncs + '\u4e00-\u9fff0-9]')
    total = len(corpus_data)
    for i, sentence in enumerate(corpus_data):
        print('Checking original data %.2f %%' % ((i+1)/total*100), end='\r')
        if len(sentence) == 0:
            # empty line
            continue
        sentence = strQ2B(sentence).replace('\r', '')
        have_ch_character = sum([1 for c in sentence if _is_chinese(c)])
        others = len(sentence) - have_ch_character
        if have_ch_character <= 3 or others >= have_ch_character:
            # too few chinese characters
            # print('Dropping sentence: '+sentence, end='\r')
            drop += 1
            continue
        # check first character is not chinese
        # check ' pair
        # check " pair
        if not _is_chinese(sentence[0]) \
           or (sum([1 for ch in sentence if ch == '"']) % 2 != 0) \
           or (sum([1 for ch in sentence if ch == '\'']) % 2 != 0) :
            #print('Dropping sentence: '+sentence, end='\r')
            drop += 1
        if strict:
            if others_re.search(sentence):
                #print('Dropping sentence: '+sentence, end='\r')
                drop += 1
                continue
        new_corpus_data.add(sentence.replace(' ', ''))
    if len(new_corpus_data) != len(corpus_data):
        if replace:
            # unclean, recreat corpus data
            data = '\n'.join(new_corpus_data)
            data = gzip.compress(data.encode())
            new_file_name = os.path.splitext(os.path.basename(data_path))
            new_file_name = new_file_name[0]+'_new'+new_file_name[1]
            with open(new_file_name, 'wb') as f:
                f.write(data)
        info = '\nClean origin corpus: {}, drop sentence: {}, left sentence: {}'.\
            format(data_path, drop, len(new_corpus_data))
        print(info)
    else:
        info = '\nOrigin corpus is clean: {}'.format(len(new_corpus_data))
        print(info)
    #with open('./log', 'a') as log:
    #    log.write('\n')
    #    log.write(info)
    return list(new_corpus_data)


def normalize_corpus_data(corpus_data,
                        normalize_char=True, normalize_digits=True,
                        normalize_punctuation=True, normalize_others=True):
    """ Args:
            corpus_data: the original corpus data in list format
            normalize_digits: if true, all digits are replaced by 0
            normalize_char: if true, all chars are replaced by a
            normalize_punctuation: if true, all punctuations are replaced by .
            normalize_others: if true, all chars(except digit,punc,ch)
                              will be replaced by o
                NOTE: this can only be set true when all others are set true.
        Return:
            processed data in tuple format
    """
    p_corpus_data = []
    for i, one in enumerate(corpus_data):
        print('Normalizing data... %.2f%%' % ((i+1)/len(corpus_data)*100), end='\r')
        one = _DIGIT_RE.sub('0', one) if normalize_digits else one
        one = _CHAR_RE.sub('a', one) if normalize_char else one
        one = _PUNC_RE.sub('.', one) if normalize_punctuation else one
        one = _OTHERS_RE.sub('o', one) if normalize_others else one
        p_corpus_data.append(one)
    print('Normalizing data...  done.')
    return tuple(p_corpus_data)


def cut_word_tokenizer(sentence, full=True):
    return list(jieba.cut(sentence, cut_all=full))


def create_vocabulary(corpus_data, max_vocabulary_size, tokenizer=cut_word_tokenizer):
    """ create vocabulary from data file(contain one sentence per line)

        Args:
            corpus_data: in list format
            max_vocabulary_size: limit on the size of the created vocabulary
            tokenizer: a function to use to tokenize each data sentence
       Returns:
            a list that contains all vocabulary
    """
    vocab = {}
    tokenized_data = []

    for i, line in enumerate(corpus_data):
        print("Creating vocabulary... %.2f%%" % ((i+1)/len(corpus_data)*100), end='\r')
        tokens = tokenizer(line)
        tokenized_data.append(tokens)
        for word in tokens:
            if not word:
                continue
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
            #for ch in word:
            #    if ch in vocab:
            #        vocab[ch] += 1
            #    else:
            #        vocab[ch] = 1
    print('Creating vocabulary...  done.')
    vocab_list = pre_vocab + sorted(vocab, key=vocab.get, reverse=True)
    if max_vocabulary_size == -1:
        to_save_vocab_list = vocab_list
    elif len(vocab_list) > max_vocabulary_size:
        to_save_vocab_list = vocab_list[:max_vocabulary_size]
    else:
        to_save_vocab_list = vocab_list
    return to_save_vocab_list, vocab, tokenized_data

def drop_unk_sentences(split_corpus_data, vocab):
    if os.path.exists('after_drop_corpus.pkl'):
        with open('after_drop_corpus.pkl', 'rb') as f:
            new_split_corpus_data = pickle.load(f)
        return new_split_corpus_data
    new_split_corpus_data = []
    drop = 0
    total = len(split_corpus_data)
    #unk_sentence_indexes = []
    for i, sentence in enumerate(split_corpus_data):
        print('deleting sentences %.2f %%' % ((i+1)/total*100), end='\r')
        words = sentence.split()
        cutted_words, unk_words = cutting_words_by_vocab(words, vocab)
        if not unk_words:
            new_split_corpus_data.append(sentence)
        else:
            drop += 1
            #unk_sentence_indexes.append(i)
#    to_insert_unk_sentence_num = min(int(total*15/100), len(unk_sentence_indexes))
#    for i in range(to_insert_unk_sentence_num):
#        r_index = random.randrange(0, len(new_split_corpus_data))
#        new_split_corpus_data.insert(r_index, split_corpus_data[unk_sentence_indexes[i]])

    info = '\nDelete unk sentence, unk: {} {} %%, result: {}'.\
            format(drop, drop//total*100, len(new_split_corpus_data))
    with open('log', 'a') as log:
        log.write('\n')
        log.write(info)
    print(info)
    with open('after_drop_corpus.pkl', 'wb') as f:
        pickle.dump(new_split_corpus_data, f)
    return new_split_corpus_data


def cutting_words_by_vocab(words, vocab):

    cutted_words = []
    unk_words = []
    for word in words:
        if word in vocab:
            cutted_words.append(word)
            continue
        for i in range(len(word)):
            index = len(vocab) + 1
            length = 0
            # to find the most long sub word in the vocab
            for i, one in enumerate(vocab):
                if one in word and len(one) > length:
                    length = len(one)
                    index = i
            if index != len(vocab) + 1:
                # found in vocab
                cutted_words.append(vocab[index])
                word = word.replace(vocab[index], '', 1)
            else:
                # not found in vocab
                break
        if word:
            unk_words.append(word)
    if unk_words:
        with open('unk_words', 'a') as f:
            f.write(str(unk_words))
            f.write('\n')
        print(str(unk_words)+' '*20)
    return cutted_words, unk_words

