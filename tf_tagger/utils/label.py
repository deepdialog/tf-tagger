import os

import numpy as np
from sklearn.preprocessing import LabelEncoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


PAD = '<PAD>'


class Label:
    def __init__(self):
        pass

    def fit(self, y):
        label = LabelEncoder()
        tags = []
        for yy in y:
            tags += yy
        label.fit([PAD] + tags)
        self.label = label
        label_size = len(label.classes_)
        assert label_size >= 2
        self.label_size = label_size

    def transform(self, y):
        max_length = int(np.max([len(yy) for yy in y]))
        return np.array([
            self.label.transform(yy + [PAD] * (max_length - len(yy)))
            for yy in y
        ])

    def inverse_transform(self, y):
        return self.label.inverse_transform(y)