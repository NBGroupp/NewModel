#!/usr/bin/python3
#coding: utf-8


from flask import Flask, jsonify, render_template, request
import tensorflow as tf

from model import run_epoch
from __init__ import app, model, session, vocab, vocab_d, nwd, UNK_INDEX


def get_one_result(model, session, vocab, data):
    TEST_STEP_SIZE = 1
    TEST_BATCH_SIZE = 1
    res = run_epoch(
        session, model, data, tf.no_op(), False,
        TEST_BATCH_SIZE, TEST_STEP_SIZE, vocab, False, False, False
    )
    return res

def proofread(sentence, model, session, vocab, vocab_d, nwd):
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
            input_data = ([pres], [lats], [candidates], target_ch)
            res = get_one_result(model, session, vocab, input_data)
            result += vocab[int(res)]
        else:
            result += char
    return result

@app.route('/correct/', methods=['GET'])
def index():
    input_sentence = request.args.get('sentence')
    if not input_sentence:
        return render_template('index.html')
    result = proofread(input_sentence, model, session, vocab, vocab_d, nwd)
    print('correctint: %s -> %s' % (input_sentence, result))
    return jsonify({'sentence': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
