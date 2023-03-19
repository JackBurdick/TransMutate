import tensorflow as tf
from tensorflow.keras.layers import Layer
from .components.compatibility.scoring_functions import ScaledDotProduct
from .components.selection.selection import SoftMaxMatMul

"""

"""


class Attention(Layer):
    """Base class for common attention wrappers

    TODO functionality:
        There is not masking functionality (yet)

    TODO:
        check issubclass(self.compatibility, ScoringFunction)
        check issubclass(self.select_and_apply, SelectAndApply)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.compatibility = None
        self.select_and_apply = None

    def call(self, q, k, v):
        # NOTE: typically the query=target, and the key/values=source

        # compute the compatibility scores
        if not self.compatibility:
            raise ValueError("Must specify a `compatibility` function")
        attn_w = self.compatibility(source=k, target=q)

        # select and apply attention scores to the values
        if not self.select_and_apply:
            raise ValueError("Must specify a `select_and_apply` function")
        out = self.select_and_apply(score_sequence=attn_w, sequence=v)

        return out


class ScaledDotProdSoftMaxMM(Attention):
    """Standard scaled dot product --> softmax and matmul"""

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.compatibility = ScaledDotProduct()
        self.select_and_apply = SoftMaxMatMul(axis=self.axis)
