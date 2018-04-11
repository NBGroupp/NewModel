#!/usr/bin/python3
#coding: utf-8


from flask import Flask, jsonify, render_template, request
import tensorflow as tf

from test_combine_cset import test
from __init__ import app, vocab, vocab_d, nwd, UNK_INDEX, count

def get_one_result(data):
    res = test(data)
    return res

def proofread(sentence, count, vocab, vocab_d, nwd):
    result = ''
    for pos, char in enumerate(sentence):
        if char in nwd and char in vocab_d:
            pres = ['START'] + [pre_c for pre_c in sentence[:pos]]
            lats = [lat_c for lat_c in sentence[pos+1:]] + ['END']
            lats.reverse()
            pres = [str(vocab_d.get(_, UNK_INDEX)) for _ in pres]
            lats = [str(vocab_d.get(_, UNK_INDEX)) for _ in lats]
            candidates = [str(vocab_d.get(_)) for _ in nwd[char]]
            target_ch = [str(vocab_d.get(char))]
            #input_data = (pres, lats, candidates, target_ch)
            input_data = (target_ch, [pres], [lats], [candidates], count)
            res = get_one_result(input_data)
            result += vocab[int(res)]
        else:
            result += char
    return result

@app.route('/correct/', methods=['GET'])
def index():
    input_sentence = request.args.get('sentence')
    if not input_sentence:
        return render_template('index.html')
    result = proofread(input_sentence, count, vocab, vocab_d, nwd)
    print('correctint: %s -> %s' % (input_sentence, result))
    return jsonify({'sentence': result})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
