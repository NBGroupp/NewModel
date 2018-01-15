#!/usr/bin/python3
#coding: utf-8


import os
from os.path import splitext
from os.path import basename
from os.path import exists
from os.path import join

from data_util import *


VOCAB_CORPUS_PATH = './corpuses/baidu-wiki.gz'
DATA_CORPUS_PATH = './corpuses/aihanyu.gz'
VOCAB_SIZE = 100000

SEG_MODEL_PATH = './ltp_data_v3.4.0/cws.model'
TAG_MODEL_PATH = './ltp_data_v3.4.0/pos.model'

from pyltp import Segmentor, Postagger
segmentor = Segmentor()
segmentor.load(SEG_MODEL_PATH)
postagger = Postagger()
postagger.load(TAG_MODEL_PATH)

name_tag = 'nh'
place_tag = 'ns'
name_replace_ch = 'N'
place_replace_ch = 'P'


def get_preprocessed_data(corpus_path):

    """ clean(reverse en-ch chars, punc, digits) and normalize(digits, chars) """

    corpus_name = splitext(basename(corpus_path))[0]
    if os.path.exists('normalized_' +  corpus_name + '.pkl'):
        print('Loading normalized data from {}...'.format(
            'normalized_'+corpus_name+'.pkl'))
        with open('normalized_' + corpus_name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        return corpus_name, data
    print('normalizing data...')

    corpus_data = clean_corpus(corpus_path, strict=True)
    corpus_data = normalize_corpus_data(corpus_data, normalize_char=True,
        normalize_digits=True, normalize_punctuation=False, normalize_others=False)

    with open('normalized_'+corpus_name+'.pkl', 'wb') as f:
        pickle.dump(corpus_data, f)

    return corpus_name, corpus_data


def filter_name_place_tokenizer(sentence):

    words = list(segmentor.segment(sentence))
    tags = list(postagger.postag(words))

    while name_tag in tags:
        words[tags.index(name_tag)] = name_replace_ch
        tags[tags.index(name_tag)] = None

    while place_tag in tags:
        words[tags.index(place_tag)] = place_replace_ch
        tags[tags.index(place_tag)] = None

    return words


def create_vocabularys(corpus_name, corpus_data, vocab_size):

    origin_vocab_name = corpus_name+'_origin_vocab_'+str(vocab_size)
    drop_vocab_name = corpus_name+'_drop_np_vocab_'+str(vocab_size)

    if exists(origin_vocab_name+'.pkl'):
        print('Loading vocab from {}...'.format(origin_vocab_name+'.pkl'))
        with open(origin_vocab_name+'.pkl', 'rb') as f:
            origin_vocab = pickle.load(f)[0]
    else:
        origin_vocab, full_origin_vocab, _ = create_vocabulary(corpus_data, vocab_size)
        with open(origin_vocab_name+'.pkl', 'wb') as f:
            pickle.dump([origin_vocab, full_origin_vocab], f)

    if exists(drop_vocab_name+'.pkl'):
        print('Loading vocab from {}...'.format(drop_vocab_name+'.pkl'))
        with open(drop_vocab_name+'.pkl', 'rb') as f:
            drop_np_vocab = pickle.load(f)[0]
    else:
        drop_np_vocab, full_drop_np_vocab, _ = create_vocabulary(corpus_data, vocab_size, tokenizer=filter_name_place_tokenizer)
        with open(drop_vocab_name+'.pkl', 'wb') as f:
            pickle.dump([drop_np_vocab, full_drop_np_vocab], f)

    return origin_vocab, drop_np_vocab


def tokenized_train_data(train_corpus_path):

    corpus_name = splitext(basename(train_corpus_path))

    if os.path.exists('tokenized_'+corpus_name+'.pkl'):
        with open('tokenized_'+corpus_name+'.pkl', 'rb') as f:
            data = pickle.load(f)
            return data

    corpus_data = clean_corpus(train_corpus_path, strict=True)
    corpus_data = normalize_corpus_data(corpus_data, normalize_char=True,
        normalize_digits=True, normalize_punctuation=False, normalize_others=False)

    tokenized_train_corpus_data = []
    for i, sentence in enumerate(corpus_data):
        print('Tokenizing training data %.2f%%' % ((i+1)/len(corpus_data)*100), end='\r')
        tokenized_train_corpus_data.append(cut_word_tokenizer(sentence), full=False)

    with open('tokenized_'+corpus_name+'.pkl', 'wb') as f:
        pickle.dump(corpus_data, f)

    return tokenized_train_corpus_data


def test_unk_percent(tokenized_train_corpus_data, vocab):

    """ vocab is a list """

    # map vocab
    vocab = {key: vocab.index(key) for key in vocab}
    unks = dict()

    unk_token_number = 0
    total_token_number = 0

    unk_sentence_number = 0
    two_more_unk_sentence_number = 0
    total_sentence_number = 0

    for i, sentence in enumerate(tokenized_train_corpus_data):

        total_token_number += len(sentence)
        total_sentence_number += 1

        unk_num_in_sentence = sum([1 for token in sentence if vocab.get(token)])
        for token in sentence:
            if not vocab.get(token):
                if unks.get(token):
                    unks[token] += 1
                else:
                    unks[token] = 1
        unk_token_number += unk_num_in_sentence
        unk_sentence_number += (unk_num_in_sentence > 0)
        two_more_unk_sentence_number += (unk_num_in_sentence >= 2)

    print('unk token in total tokens: %d/%d  %.2f%%' %
          (unk_token_number, total_token_number, unk_token_number/total_token_number*100))
    print('unk sentence in total sentences: %d/%d  %.2f%%' %
          (unk_sentence_number, total_sentence_number, unk_sentence_number/total_sentence_number*100))
    print('2 more unks sentence number in total_sentence_number: %d/%d  %.2f%%' %
          (two_more_unk_sentence_number, total_sentence_number, two_more_unk_sentence_number/total_sentence_number*100))

    return unks

def run():

    vocab_corpus_name, vocab_corpus_data = get_preprocessed_data(VOCAB_CORPUS_PATH)
    origin_vocab, drop_np_vocab = create_vocabularys(vocab_corpus_name, vocab_corpus_data, VOCAB_SIZE)
    tokenized_train_data = tokenized_train_data(DATA_CORPUS_PATH)

    print('USING ORIGIN VOCAB:')
    origin_unks = test_unk_percent(tokenized_train_data, origin_vocab)
    with open('origin_unks.pkl', 'wb') as f:
        pickle.dump(origin_unks, f)

    print('\nUSING DROP NP VOCAB:')
    drop_unks = test_unk_percent(tokenized_train_data, drop_np_vocab)
    with open('drop_unks.pkl', 'wb') as f:
        pickle.dump(drop_unks, f)



if __name__ == '__main__':
    run()
