#!/usr/bin/python3
#coding: utf-8


import time
from os.path import splitext
from os.path import basename
from os.path import exists
from os.path import join

from data_util import *


def create_input_target_data(tokenized_data, vocab, k, unk_percent, vectorize):

    paddings = [PAD for i in range(k)]
    total_unk = len(tokenized_data) * unk_percent

    data1 = []
    data2 = []
    target = []
    current_unk = 0
    unks = {}

    for i, one in enumerate(tokenized_data):

        print('Creating input and target data... %.2f%%'
              % ((i+1)/len(tokenized_data)*100), end='\r')

        #unk_in_sentence = 1 if [1 for token in one if not vocab.get(token)] else 0
        unk_in_sentence = 0
        for token in one:
            if not vocab.get(token):
                unk_in_sentence = 1
                if not unks.get(token):
                    unks[token] = 1
                else:
                    unks[token] += 1

        current_unk += unk_in_sentence
        if unk_in_sentence and current_unk > total_unk:
            continue

        if k == -1:
            for i, ch in enumerate(one):
                if ch in puncs or ch == '0' or ch =='a' or not vocab.get(ch):
                    continue
                pres = []
                lats = []
                if i == 0:
                    pres.append(str(vocab.get(PAD)))
                else:
                    pres = [str(vocab.get(pre_c, UNK_INDEX)) for pre_c in one[:i]]

                if i == len(one)-1:
                    lats.append(str(vocab.get(PAD)))
                else:
                    lats = [str(vocab.get(lat_c, UNK_INDEX)) for lat_c in one[:i]]

                data1.append(pres)
                data2.append(lats)
                target.append(str(vocab.get(ch)))
        else:
            tokens = paddings + one + paddings
            for i, ch in enumerate(one):
                if ch in puncs or ch == '0' or ch == 'a' or not vocab.get(ch):
                    continue
                if vectorize:
                    pres = []
                    lats = []
                    pres = [str(vocab.get(pre_c, UNK_INDEX)) for pre_c in tokens[i:i+k]]
                    lats = [str(vocab.get(lat_c, UNK_INDEX)) for lat_c in tokens[i+k+1:i+k+1+k]]
                    data1.append(pres)
                    data2.append(lats)
                else:
                    data1.append(tokens[i:i+k])
                    data2.append(tokens[i+k+1:i+k+1+k])
                target.append(str(vocab.get(ch)))

    unks = sorted(unks, key=lambda x: unks[x], reverse=True)

    return data1, data2, target, current_unk, unks


def generate_data(vocab_data_path, train_data_path, max_vocabulary_size, train_sentence_number, k):

    normalize_punctuation = False
    unk_percent = 0.05

    data_dir = join(os.getcwd(), time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    if not exists(data_dir):
        os.mkdir(data_dir)
    print('data will be saved in {}'.format(data_dir))

    # preprocess vocab data
    #vocab_corpus_name = splitext(basename(vocab_data_path))[0]
    #processed_vocab_corpus_name = vocab_corpus_name +'.processed.pkl'
    #if exists(processed_vocab_corpus_name):
    #    print('Loading preprocessed corpus data from {}...'.format(processed_vocab_corpus_name))
    #    with open(processed_vocab_corpus_name, 'rb') as f:
    #        vocab_corpus_data = pickle.load(f)
    #else:
    #    vocab_corpus_data = clean_corpus(vocab_data_path, strict=True)
    #    vocab_corpus_data = normalize_corpus_data(
    #        vocab_corpus_data, normalize_char=True,
    #        normalize_digits=True, normalize_punctuation=False, normalize_others=False
    #    )
    #    with open(processed_vocab_corpus_name, 'wb') as f:
    #        pickle.dump(vocab_corpus_data, f)

    # create vocabulary
    # processed_vocab_name = vocab_corpus_name + '.vocab.' + str(max_vocabulary_size)
    processed_vocab_name = 'vocab.112801'
    if exists(processed_vocab_name):
        print('Loading vocab from {}...'.format(processed_vocab_name))
        with open(processed_vocab_name, 'r') as f:
            vocab = f.read().split('\n')
    else:
        vocab, _, _= create_vocabulary(vocab_corpus_data, max_vocabulary_size)
        with open(processed_vocab_name, 'w') as f:
            f.write('\n'.join(vocab))
    # change to map
    vocab = {key: vocab.index(key) for key in vocab}

    # tokenized train data
    train_corpus_name = splitext(basename(train_data_path))[0]
    processed_train_corpus_name = train_corpus_name + '.processed.pkl'
    if exists(processed_train_corpus_name):
        print('Loading preprocessed corpus data from {}...'.format(processed_train_corpus_name))
        with open(processed_train_corpus_name, 'rb') as f:
            tokenized_train_corpus_data = pickle.load(f)
    else:
        train_corpus_data = clean_corpus(train_data_path, strict=True)
        train_corpus_data = normalize_corpus_data(
            train_corpus_data, normalize_char=True,
            normalize_digits=True, normalize_punctuation=False, normalize_others=False
        )

        tokenized_train_corpus_data = []
        for i, sentence in enumerate(train_corpus_data):
            print('Tokenizing training data %.2f%%' % ((i+1)/len(train_corpus_data)*100), end='\r')
            tokenized_train_corpus_data.append(cut_word_tokenizer(sentence, full=False))

        with open(processed_train_corpus_name, 'wb') as f:
            pickle.dump(tokenized_train_corpus_data, f)

    # create input and targrt data
    if train_sentence_number > 0:
        data1, data2, target, current_unk, unks = create_input_target_data(
            tokenized_train_corpus_data[:train_sentence_number], vocab, k, unk_percent, vectorize=True)
    else:
        data1, data2, target, current_unk, unks = create_input_target_data(
            tokenized_train_corpus_data, vocab, k, unk_percent, vectorize=True)

    data_len = len(data1)
    # split into dev(0.2), val(0.2), train(0.6)
    two_split_index = int(data_len/10*2)
    four_split_index = int(data_len/10*4)

    # save corpus
    with open(join(data_dir, 'data1.'+str(len(data1))), 'w') as f:
        for one in data1:
            f.write(' '.join(one)+'\n')
    with open(join(data_dir, 'data2.'+str(len(data2))), 'w') as f:
        for one in data2:
            f.write(' '.join(one)+'\n')
    with open(join(data_dir, 'target.'+str(len(target))), 'w') as f:
        for one in target:
            f.write(one+'\n')
    with open(join(data_dir, 'unk.'+str(len(unks))), 'w') as f:
        f.write('\n'.join(unks))
    print('{} unk pairs in {}'.format(current_unk, sum([len(s) for s in tokenized_train_corpus_data])))


if __name__ == '__main__':
    vocab_data_path = sys.argv[1]
    train_data_path = sys.argv[2]
    max_vocabulary_size = int(sys.argv[3])
    train_sentence_number = int(sys.argv[4])
    k = int(sys.argv[5])
    generate_data(vocab_data_path, train_data_path, max_vocabulary_size, train_sentence_number, k)

