# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf

from .label import PAD, SOS, EOS, UNK

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Tokenizer:
    def __init__(self, vocab_file=None):
        self.word_index = {
            PAD: 0,
            UNK: 1,
            SOS: 2,
            EOS: 3
        }
        if vocab_file is not None:
            with open(vocab_file, 'r') as fp:
                i = 0
                for line in fp:
                    if line.endswith('\n'):
                        line = line[:-1]
                    if len(line):
                        self.word_index[line] = i
                        i += 1
            self.index_word = {v: k for k, v in self.word_index.items()}
            self.vocab_size = len(self.word_index)

    def fit(self, X):
        for sent in X:
            for word in sent:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
        self.vocab_size = len(self.word_index)
        self.index_word = {v: k for k, v in self.word_index.items()}

    def inverse_transform(self, X):
        ret = []
        for sent in X:
            words = []
            for w in sent:
                if w <= 0:
                    break
                if w in self.index_word:
                    words.append(self.index_word[w])
            ret.append(words)
        return ret

    def transform(self, X):
        max_length = max([len(x) for x in X]) + 2
        ret = []
        for sent in X:
            vec = []
            vec.append(self.word_index[SOS])
            for word in sent:
                if word in self.word_index:
                    vec.append(self.word_index[word])
                else:
                    vec.append(self.word_index[UNK])
            vec.append(self.word_index[EOS])
            if len(vec) < max_length:
                vec = vec + [self.word_index[PAD]] * (max_length - len(vec))
            ret.append(vec)
        return np.array(ret, dtype=np.int32)


if __name__ == "__main__":
    tokenizer = Tokenizer('./chinese_L-12_H-768_A-12/vocab.txt')
    print(tokenizer.transform([['[CLS]', '我', '爱', '你', '[SEP]']]))
    print(tokenizer.transform([['我', '爱', '你']]))