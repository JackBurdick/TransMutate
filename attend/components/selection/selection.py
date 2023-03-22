import tensorflow as tf
from tensorflow.keras.layers import Layer
from .components.apply import SelectionApply, MatMul
from .components.values import SelectionFunction, SoftMax

"""
The name of this isn't exactly right, but using `selection` as a placeholder for
now.
"""


class SelectAndApply(Layer):
    """Base class for specified 'selection' functions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.selection = None
        if self.selection:
            assert issubclass(
                self.selection, SelectionFunction
            ), f"{self.selection} must be a subclass of SelectionFunction"

        self.apply = None
        if self.apply:
            assert issubclass(
                self.apply, SelectionApply
            ), f"{self.apply} must be a subclass of SelectionApply"

    def call(self, score_sequence, sequence):
        if not self.selection:
            raise ValueError("Must specify a `selection` function")
        weights = self.selection(sequence=score_sequence)

        if not self.apply:
            raise ValueError("Must specify an `apply` function")
        out = self.apply(weights=weights, sequence=sequence)

        return out


class SoftMaxMatMul(SelectAndApply):
    """Standard softmax and matmul"""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.selection = SoftMax(axis=self.axis)
        self.apply = MatMul()
