#!/usr/bin/python3
#coding: utf-8


import time
import random
from os import rename
from os.path import splitext
from os.path import basename
from os.path import dirname
from os.path import exists
from os.path import join
from sys import exit

from data_util import *
from config import *


def mix_error(tokens, near_words_dict, l_error_ratio, u_error_ratio):

    nears = [token for token in tokens if near_words_dict.get(token, -1) != -1]
    if len(nears) == 0:
        return None
    if l_error_ratio >= (len(nears) / len(tokens)):
        return None

    near_num = int(len(tokens) * u_error_ratio + 0.5)

    mixxed_chars = []
    mixxed_positions = []
    targets = []
    for near in nears[:near_num]:
        to_mix = random.sample(near_words_dict[near], 1)[0]
        while near == to_mix:
            to_mix = random.sample(near_words_dict[near], 1)[0]
        mixxed_chars.append(to_mix)
        mixxed_positions.append(tokens.index(near))
        tokens[tokens.index(near)] = to_mix
        targets.append(near)

    return {
        'tokens': tokens,
        'mixxed_chars': mixxed_chars,
        'mixxed_positions': mixxed_positions,
        'targets': targets
    }


def create_input_target_data(data_dir, train_corpus_data, tokenizer,
                             near_words_dict, l_error_ratio, u_error_ratio,
                             vocab, k, max_unk_percent_in_sentence, vectorize,
                             unk_percent=1, operate_in_file=False):

    if k >= 0:
        print('not implement fixed method')
        exit(1)
    if not operate_in_file:
        print('must select operate in file')
        exit(1)

    paddings = [PAD for i in range(k)]

    data1 = []
    data2 = []
    target = []
    data_mixxed = []
    data_nears = []
    data_sum = 0
    data_unk_sum = 0
    unks = {}

    for i, sentence in enumerate(train_corpus_data):

        print('Creating input and target data... %.2f%%, current: %d data from %d sentences'
              % ((i+1)/len(train_corpus_data)*100, data_sum, i+1), end='\r')

        # tokenize
        _tokens = tokenizer(sentence)
        mixed = mix_error(_tokens, near_words_dict, l_error_ratio, u_error_ratio)
        if not mixed:
            continue
        else:
            one = mixed['tokens']
            mixxed_chars = mixed['mixxed_chars']
            mixxed_positions = mixed['mixxed_positions']
            targets_chars = mixed['targets']

        for token in one:
            if token not in vocab:
                if token not in unks:
                    unks[token] = 1
                else:
                    unks[token] += 1
        unk_num_in_sentence = sum([1 for token in one if vocab.get(token, -1) == -1])
        unk_in_sentence = 1 if unk_num_in_sentence > 0 else 0
        if unk_num_in_sentence / len(one) > max_unk_percent_in_sentence:
            continue
        #if unk_in_sentence:  # NOTICE: drop all unks
        #    continue

        if data_sum \
           and unk_num_in_sentence and data_unk_sum / data_sum > unk_percent:
            continue

        for ch, t, position in zip(mixxed_chars, targets_chars, mixxed_positions):
            pres = []
            lats = []
            ch_near_words = [_ for _ in near_words_dict[ch]]

            if k == -1:
                # unfixed
                pres = ['START'] + [pre_c for pre_c in one[:position]]
                lats = [lat_c for lat_c in one[position+1:]] + ['END']
            else:
                # fixed
                tokens = paddings + one + paddings
                pres = [pre_c for pre_c in tokens[position:position+k]]
                lats = [lat_c for lat_c in tokens[position+k+1:position+k+1+k]]

            lats.reverse()

            target_ch = t
            mixxed_ch = ch
            if vectorize:
                pres = [str(vocab.get(_, UNK_INDEX)) for _ in pres]
                lats = [str(vocab.get(_, UNK_INDEX)) for _ in lats]
                ch_near_words = [str(vocab.get(_)) for _ in ch_near_words]
                target_ch = str(vocab.get(target_ch))
                mixxed_ch = str(vocab.get(mixxed_ch))
            data1.append(pres)
            data2.append(lats)
            target.append(target_ch)
            data_nears.append(ch_near_words)
            data_mixxed.append(mixxed_ch)
            data_sum += 1
            if unk_in_sentence:
                data_unk_sum += 1
            if operate_in_file and len(data1) % 100000 == 0:
                with open(join(data_dir, 'data1'), 'a') as f:
                    for d in data1:
                        f.write(' '.join(d) + '\n')
                with open(join(data_dir, 'data2'), 'a') as f:
                    for d in data2:
                        f.write(' '.join(d) + '\n')
                with open(join(data_dir, 'target'), 'a') as f:
                    for t in target:
                        f.write(t + '\n')
                with open(join(data_dir, 'nears'), 'a') as f:
                    for n in data_nears:
                        f.write(' '.join(n) + '\n')
                with open(join(data_dir, 'error_origins'), 'a') as f:
                    for m in data_mixxed:
                        f.write(m + '\n')
                del data1
                del data2
                del target
                del data_nears
                del data_mixxed
                data1 = []
                data2 = []
                target = []
                data_nears = []
                data_mixxed = []
            del pres
            del lats

    if operate_in_file and len(data1) > 0:
        with open(join(data_dir, 'data1'), 'a') as f:
            for d in data1:
                f.write(' '.join(d) + '\n')
        with open(join(data_dir, 'data2'), 'a') as f:
            for d in data2:
                f.write(' '.join(d) + '\n')
        with open(join(data_dir, 'target'), 'a') as f:
            for t in target:
                try:
                    f.write(t + '\n')
                except:
                    pass
        with open(join(data_dir, 'nears'), 'a') as f:
            for n in data_nears:
                f.write(' '.join(n) + '\n')
        with open(join(data_dir, 'error_origins'), 'a') as f:
            for m in data_mixxed:
                f.write(m + '\n')
        rename(join(data_dir, 'data1'), join(data_dir, 'data1.'+str(data_sum)))
        rename(join(data_dir, 'data2'), join(data_dir, 'data2.'+str(data_sum)))
        rename(join(data_dir, 'target'), join(data_dir, 'target.'+str(data_sum)))
        rename(join(data_dir, 'nears'), join(data_dir, 'nears.'+str(data_sum)))
        rename(join(data_dir, 'error_origins'), join(data_dir, 'error_origins.'+str(data_sum)))

    with open(join(data_dir, 'unks.pkl'), 'wb') as f:
        pickle.dump(unks, f)


    if operate_in_file:
        return data_sum, data_unk_sum
    else:
        return data1, data2, target


def generate_data(data_dir):

    normalize_punctuation = False

    vocab_corpus_name = splitext(basename(VOCAB_DATA_PATH))[0]
    vocab_corpus_path = dirname(VOCAB_DATA_PATH)
    train_corpus_name = splitext(basename(TRAIN_DATA_PATH))[0]
    train_corpus_path = dirname(TRAIN_DATA_PATH)

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
    if VOCAB_FILE:
        print('Loading vocab file from {}...'.format(VOCAB_FILE))
        with open(VOCAB_FILE, 'r') as f:
            vocab = f.read().split('\n')
    else:
        normalized_vocab_corpus_name = vocab_corpus_name +'.normalized.pkl'
        normalized_vocab_corpus_name = join(vocab_corpus_path, normalized_vocab_corpus_name)
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
        processed_vocab_name = join(vocab_corpus_path, processed_vocab_name)
        if exists(processed_vocab_name):
            print('Loading vocab from {}...'.format(processed_vocab_name))
            with open(processed_vocab_name, 'r') as f:
                vocab = f.read().split('\n')
        else:
            vocab, _, _= create_vocabulary(vocab_corpus_data, VOCAB_SIZE, tokenizer=VOCAB_TOKENIZER)
            with open(processed_vocab_name, 'w') as f:
                f.write('\n'.join(vocab))
    # save vocab
    with open(join(data_dir, 'vocab.'+str(len(vocab))), 'w') as f:
        f.write('\n'.join(vocab))

    # change to map
    vocab = {key: vocab.index(key) for key in vocab}

    # build near words dict
    print('Creating near words dict...', end='\r')
    with open(NEAR_WORDS_DATA_PATH, 'r') as f:
        near_words_data = f.read().strip().split('\n')
    near_words_dict = {}
    drop = 0
    for words in near_words_data:
        chs = words.strip().split()
        chs = [_ for _ in chs if vocab.get(_)]
        if len(chs) <= 1:
            drop += 1
            continue
        #if sum([1 for ch in chs if vocab.get(chs, -1) == -1]) > 0:
        #    # 含有UNK，跳过该组
        #    drop += 1
        #    continue
        for ch in chs:
            if not near_words_dict.get(ch):
                near_words_dict[ch] = chs.copy()
            else:
                near_words_dict[ch] += [c for c in chs if c not in near_words_dict[ch]]
    print('Creating near words dict... done. drop %d/%d groups near words, %d keys'
          % (drop, len(near_words_data), len(near_words_dict.keys())))
    with open('near_words_dict.pkl', 'wb') as f:
        pickle.dump(near_words_dict, f)

    # process train data
    normalized_train_corpus_name = train_corpus_name +'.normalized.pkl'
    normalized_train_corpus_name = join(train_corpus_path, normalized_train_corpus_name)
    if exists(normalized_train_corpus_name):
        with open(normalized_train_corpus_name, 'rb') as f:
            train_corpus_data = pickle.load(f)
    else:
        train_corpus_data = clean_corpus(TRAIN_DATA_PATH, strict=True)
        train_corpus_data = normalize_corpus_data(
            train_corpus_data, normalize_char=True,
            normalize_digits=True, normalize_punctuation=False, normalize_others=False
        )
        with open(normalized_train_corpus_name, 'wb') as f:
            pickle.dump(train_corpus_data, f)

    # create input and targrt data
    if OPERATE_IN_FILE:
        data_sum, data_unk_sum = create_input_target_data(
            data_dir, train_corpus_data, TRAIN_DATA_TOKENIZER, near_words_dict,
            LOW_ERROR_RATIO, UP_ERROR_RATIO, vocab, K, MAX_UNK_PERCENT_IN_SENTENCE,
            VECTORIZE, unk_percent=UNK_PERCENT_IN_TOTAL_DATA, operate_in_file=OPERATE_IN_FILE)
        print('\nTotal data: {}, unk data: {}, unk percent: {:.2f}%%'.\
              format(data_sum, data_unk_sum, data_unk_sum / (data_sum+1) * 100))
    else:
        data1, data2, target = create_input_target_data(
            data_dir, train_corpus_data, TRAIN_DATA_TOKENIZER, near_words_dict, vocab, K,
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


if __name__ == '__main__':
    data_dir = join(os.getcwd(), time.strftime('%y.%m.%d-%H:%M', time.localtime(time.time())))
    if not exists(data_dir):
        os.mkdir(data_dir)
    generate_data(data_dir)
