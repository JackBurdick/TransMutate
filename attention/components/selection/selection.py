import tensorflow as tf
from tensorflow.keras.layers import Layer
from .components.apply import MatMul
from .components.values import SoftMax

"""
The name of this isn't exactly right, but using `selection` as a placeholder for
now.
"""


class SelectAndApply(Layer):
    """Base class for specified 'selection' functions

    TODO:
        check issubclass(self.apply, SelectionApply)
        check issubclass(self.selection, SelectionFunction)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selection = None
        self.apply = None

    def call(self, score_sequence, sequence):
        if not self.selection:
            raise ValueError("Must specify a `selection` function")
        weights = self.selection(sequence=score_sequence)

        if not self.apply:
            raise ValueError("Must specify a `apply` function")
        out = self.apply(weights=weights, sequence=sequence)

        return out


class SoftMaxMatMul(SelectAndApply):
    """Standard softmax and matmul"""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.selection = SoftMax(axis=self.axis)
        self.apply = MatMul()
