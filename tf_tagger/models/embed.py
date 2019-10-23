
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa


class Embed(tf.keras.Model):

    def __init__(self, embedding_size, vocab_size):
        super(Embed, self).__init__(self)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(
            vocab_size,
            embedding_size
        ))
        self.model = model

    @tf.function
    def call(self, inputs):
        return self.model(inputs)

