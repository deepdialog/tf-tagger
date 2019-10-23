
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from .embed import Embed
from .encoder import Encoder
from .decoder import Decoder

class TaggerModel(tf.keras.Model):
    def __init__(self,
                 embedding_size,
                 vocab_size,
                 tag_size):
        super(TaggerModel, self).__init__(self)
        self.emb = Embed(
            embedding_size=embedding_size,
            vocab_size=vocab_size
        )
        self.en = Encoder(
            embedding_size=embedding_size
        )
        self.project = tf.keras.models.Sequential([
            tf.keras.layers.Dense(tag_size)
        ])
        self.project.build(input_shape=(None, embedding_size * 2))
        self.de = Decoder(
            tag_size=tag_size
        )

    @tf.function
    def call(self, inputs):
        lengths = tf.reduce_sum(tf.cast(tf.math.greater(inputs, 0), tf.int32), axis=-1)
        m = self.emb(inputs)
        m = self.en(m)
        m = self.project(m)
        m = self.de(m, lengths)
        return m

    @tf.function
    def compute_loss(self, inputs, tags):
        lengths = tf.reduce_sum(tf.cast(tf.math.greater(inputs, 0), tf.int32), axis=-1)
        m = self.emb(inputs)
        m = self.en(m)
        m = self.project(m)
        return self.de.compute_loss(m, lengths, tags)

