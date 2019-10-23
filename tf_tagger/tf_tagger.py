import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .models.tagger_model import TaggerModel
from .utils.label import Label
from .utils.tokenizer import Tokenizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TFTagger:
    def __init__(self, embedding_size=100, batch_size=32, epoch=100):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = None

    def build_model(self):
        return TaggerModel(embedding_size=self.embedding_size,
                           vocab_size=self.tokenizer.vocab_size,
                           tag_size=self.label.label_size)

    def fit(self, X, y):
        """Model training."""

        tokenizer = Tokenizer()
        tokenizer.fit(X)
        self.tokenizer = tokenizer

        label = Label()
        label.fit(y)
        self.label = label

        if self.model is None:
            model = self.build_model()
            self.model = model
        else:
            model = self.model

        optimizer = tf.keras.optimizers.Adam()

        X_vec = tokenizer.transform(X)
        y_vec = label.transform(y)

        total_batch = int(np.ceil(len(X_vec) / self.batch_size))
        for i_epoch in range(self.epoch):
            pbar = tqdm(range(total_batch), ncols=100)
            pbar.set_description(f'epoch: {i_epoch} loss: /')
            for i in pbar:
                i_min = i * self.batch_size
                i_max = min((i + 1) * self.batch_size, len(X_vec))
                x = X_vec[i_min:i_max]
                tags = y_vec[i_min:i_max]
                x = tf.convert_to_tensor(x, dtype=tf.int32)
                tags = tf.convert_to_tensor(tags, dtype=tf.int32)
                with tf.GradientTape() as tape:
                    loss = model.compute_loss(x, tags)
                    gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))
                loss = loss.numpy().sum()

                pbar.set_description(f'epoch: {i_epoch} loss: {loss:.4f}')

    def predict(self, X):
        """Predict label."""
        assert self.model is not None, 'Intent not fit'
        x = self.tokenizer.transform(X)
        x = tf.convert_to_tensor(x, dtype=tf.int32)
        x = self.model(x)
        x = x.numpy()
        x = [
            self.label.inverse_transform(xx)[:len(X[i])].tolist()
            for i, xx in enumerate(x)
        ]
        return x

    def __getstate__(self):
        """Pickle compatible."""
        state = self.__dict__.copy()
        if self.model is not None:
            state['model_weights'] = state['model'].get_weights()
            del state['model']
        return state

    def __setstate__(self, state):
        """Pickle compatible."""
        if 'model_weights' in state:
            model_weights = state.get('model_weights')
            del state['model_weights']
            self.__dict__.update(state)
            self.model = self.build_model()
            self.model.set_weights(model_weights)
        else:
            self.__dict__.update(state)
