import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
The name of this isn't exactly right, but using `selection` as a placeholder for
now. This file will contain a "selection" function (typically a softmax)
"""


class SelectionFunction(Layer):
    """Base class for specified 'selection' functions"""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.selection_fn = self._selection_fn

    def _selection_fn(self, sequence):
        raise NotImplementedError(
            "Must implement a `_selection_fn(sequence)` in the subclass"
        )

    def call(self, sequence):
        selection_scores = self.selection_fn(sequence=sequence)
        return selection_scores


class SoftMax(SelectionFunction):
    """Standard softmax"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selection_fn = self.softmax

    def softmax(self, sequence):
        """apply softmax over sequence

        sequence is commonly the attention weights
        """
        out = tf.nn.softmax(sequence, axis=self.axis)
        return out