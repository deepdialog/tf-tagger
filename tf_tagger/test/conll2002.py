import os
import pickle
import nltk
from ..tf_tagger import TFTagger


def test():

    train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
    # test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))

    it = TFTagger(
        batch_size=32,
        embedding_size=100
    )
    x = [
        [xxx[0] for xxx in xx] for xx in train_sents
    ]
    y = [
        [xxx[2] for xxx in xx] for xx in train_sents
    ]

    it.fit(x, y)
    print(it.predict(x))
    with open('/tmp/test.model', 'wb') as fp:
        pickle.dump(it, fp)
    with open('/tmp/test.model', 'rb') as fp:
        it = pickle.load(fp)
    print(it.predict(x))


if __name__ == '__main__':
    test()
