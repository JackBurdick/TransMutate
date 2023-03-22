import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
The name of this isn't exactly right, but using `selection` as a placeholder for
now. This file will contain the function for "applying" a selection
score/weights to a sequence (typically the values in standard attention)
"""


class SelectionApply(Layer):
    """Base class for specified 'selection' functions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_fn = self._apply_fn

    def _apply_fn(self, weights, sequence):
        raise NotImplementedError(
            "Must implement a `_apply_fn(weights, sequence)` in the subclass"
        )

    def call(self, weights, sequence):
        out = self.apply_fn(weights=weights, sequence=sequence)
        return out


class MatMul(SelectionApply):
    """Standard matmul"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_fn = self.matmul

    def matmul(self, weights, sequence):
        """apply weights to the sequence

        weights is commonly the attention weights and sequence is commonly the
        values
        """
        out = tf.matmul(weights, sequence)
        return out