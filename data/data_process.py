#!/usr/bin/python3
#coding: utf-8


import time
from os.path import splitext
from os.path import basename
from os.path import exists
from os.path import join

from data_util import *
from config import *


def create_input_target_data(tokenized_data, vocab, k,
                             max_unk_percent_in_sentence, vectorize=True, unk_percent=1):

    paddings = [PAD for i in range(k)]
    total_unk = len(tokenized_data) * unk_percent

    data1 = []
    data2 = []
    target = []
    unk_data1 = []
    unk_data2 = []
    unk_target = []

    for i, one in enumerate(tokenized_data):

        print('Creating input and target data... %.2f%%'
              % ((i+1)/len(tokenized_data)*100), end='\r')

        unk_num_in_sentence = sum([1 for token in one if not vocab.get(token)])
        unk_in_sentence = 1 if unk_num_in_sentence > 0 else 0
        if unk_num_in_sentence / len(one) > max_unk_percent_in_sentence:
            continue
        #if unk_in_sentence:  # NOTICE: drop all unks
        #    continue

        if k == -1:  # not implement whether vectorize
            for i, ch in enumerate(one):
                if ch in puncs or ch == '0' or ch =='a' or not vocab.get(ch) or ch=='N' or ch =='P':
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
                    lats = [str(vocab.get(lat_c, UNK_INDEX)) for lat_c in one[i+1:]]

                if not unk_in_sentence:
                    data1.append(pres)
                    data2.append(lats)
                    target.append(str(vocab.get(ch)))
                else:
                    unk_data1.append(pres)
                    unk_data2.append(lats)
                    unk_target.append(str(vocab.get(ch)))
        else:  # not implement mixing unk
            tokens = paddings + one + paddings
            for i, ch in enumerate(one):
                if ch in puncs or ch == '0' or ch == 'a' or not vocab.get(ch) or ch=='N' or ch=='P':
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

    if k == -1:  # mixing unk
        print('Mixing unks...', end='\r')
        unk_num = int(len(data1) * unk_percent)
        if unk_num > len(unk_data1):  # lack unk datas
            unk_indices = list(range(len(unk_data1)))
        else:
            unk_indices = random.sample(range(len(unk_data1)), unk_num)
        print('no unk data: {}, to mix unk data: {}'.format(len(data1), len(unk_indices)))
        for index in unk_indices:
            insert_index = random.randrange(0, len(data1))
            data1.insert(insert_index, unk_data1[index])
            data2.insert(insert_index, unk_data2[index])
            target.insert(insert_index, unk_target[index])
        print('Mixing {} unks into {} data'.format(len(unk_indices), len(data1)))

    return data1, data2, target


def generate_data():

    normalize_punctuation = False

    vocab_corpus_name = splitext(basename(VOCAB_DATA_PATH))[0]
    train_corpus_name = splitext(basename(TRAIN_DATA_PATH))[0]
    data_dir = join(os.getcwd(), time.strftime('%y.%m.%d-%H:%M', time.localtime(time.time())))
    if not exists(data_dir):
        os.mkdir(data_dir)
    # log info
    with open(join(data_dir, 'log'), 'w') as f:
        f.write('Vocab builded from:' + vocab_corpus_name + '\n')
        f.write('Train data builded from: ' + train_corpus_name + '\n')
        f.write('max_vocabulary_size: ' + str(VOCAB_SIZE) + '\n')
        f.write('k: ' + str(K) + '\n')
        f.write('max unk percent in one sentence: ' + str(MAX_UNK_PERCENT_IN_SENTENCE) + '\n')
        f.write('unk percent in total data({}): {}\n'.format(
            'if 1, means that all unks mixed into data, not whole data contain unk',
            str(UNK_PERCENT_IN_TOTAL_DATA)))
    print('data will be saved in {}'.format(data_dir))

    # preprocess vocab data
    normalized_vocab_corpus_name = vocab_corpus_name +'.normalized.pkl'
    if exists(normalized_vocab_corpus_name):
        print('Loading preprocessed corpus data from {}...'.format(normalized_vocab_corpus_name))
        with open(normalized_vocab_corpus_name, 'rb') as f:
            vocab_corpus_data = pickle.load(f)
    else:
        vocab_corpus_data = clean_corpus(VOCAB_DATA_PATH, strict=True)
        vocab_corpus_data = normalize_corpus_data(
            vocab_corpus_data, normalize_char=True,
            normalize_digits=True, normalize_punctuation=False, normalize_others=False
        )
        with open(normalized_vocab_corpus_name, 'wb') as f:
            pickle.dump(vocab_corpus_data, f)

    # create vocabulary
    processed_vocab_name = vocab_corpus_name + '.vocab.' + str(VOCAB_SIZE)
    if exists(processed_vocab_name):
        print('Loading vocab from {}...'.format(processed_vocab_name))
        with open(processed_vocab_name, 'r') as f:
            vocab = f.read().split('\n')
    else:
        vocab, _, _= create_vocabulary(vocab_corpus_data, VOCAB_SIZE, tokenizer=VOCAB_TOKENIZER)
        with open(processed_vocab_name, 'w') as f:
            f.write('\n'.join(vocab))
    # change to map
    vocab = {key: vocab.index(key) for key in vocab}

    # tokenized train data
    tokenized_train_corpus_name = train_corpus_name + '.tokenized.pkl'
    if exists(tokenized_train_corpus_name):
        print('Loading tokenized corpus data from {}...'.format(tokenized_train_corpus_name))
        with open(tokenized_train_corpus_name, 'rb') as f:
            tokenized_train_corpus_data = pickle.load(f)
    else:
        if exists(train_corpus_name + '.normalized.pkl'):
            with open(train_corpus_name + '.normalized.pkl', 'rb') as f:
                train_corpus_data = pickle.load(f)
        else:
            train_corpus_data = clean_corpus(TRAIN_DATA_PATH, strict=True)
            train_corpus_data = normalize_corpus_data(
                train_corpus_data, normalize_char=True,
                normalize_digits=True, normalize_punctuation=False, normalize_others=False
            )
            with open(train_corpus_name + '.normalized.pkl', 'wb') as f:
                pickle.dump(train_corpus_data, f)

        tokenized_train_corpus_data = []
        for i, sentence in enumerate(train_corpus_data):
            print('Tokenizing training data %.2f%%' % ((i+1)/len(train_corpus_data)*100), end='\r')
            tokenized_train_corpus_data.append(TRAIN_DATA_TOKENIZER(sentence))

        with open(tokenized_train_corpus_name, 'wb') as f:
            pickle.dump(tokenized_train_corpus_data, f)

    # create input and targrt data
    data1, data2, target = create_input_target_data(
        tokenized_train_corpus_data, vocab, K,
        MAX_UNK_PERCENT_IN_SENTENCE, unk_percent=UNK_PERCENT_IN_TOTAL_DATA)

    # save corpus
    with open(join(data_dir, 'data1.'+str(len(data1))), 'w') as f:
        for one in data1:
            f.write(' '.join(one)+'\n')
    with open(join(data_dir, 'data2.'+str(len(data2))), 'w') as f:
        for one in data2:
            f.write(' '.join(one)+'\n')
    with open(join(data_dir, 'target.'+str(len(target))), 'w') as f:
        f.write('\n'.join(target))

generate_data()
