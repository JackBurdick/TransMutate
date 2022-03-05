import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer

"""
Create values for positional information
"""


class PositionalValues(Layer):
    """Base class for specified positional value functions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.values_fn = self._values_fn

    def _values_fn(self):
        raise NotImplementedError(f"Must implement a `values_fn()` in the subclass")

    def call(self):
        out = self.values_fn()
        return out


class EncodingValues(PositionalValues):
    """Base class for creating absolute positional values/encodings"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.values_fn = self._values_fn

    def _values_fn(self, seq_len, embed_d):
        raise NotImplementedError(
            f"Must implement a `values_fn(seq_len, embed_d)` in the subclass"
        )

    def call(self, seq_len, embed_d):
        out = self.values_fn(seq_len, embed_d)
        return out


class Sinusoidal(EncodingValues):
    """Standard positional encoding from original paper

    The core functionality is derived from:
        - https://www.tensorflow.org/text/tutorials/transformer#positional_encoding

    @article{Vaswani2017AttentionIA,
        title={Attention is All you Need},
        author={Ashish Vaswani and Noam M. Shazeer and Niki Parmar
                and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez
                and Lukasz Kaiser and Illia Polosukhin},
        journal={ArXiv},
        year={2017},
        volume={abs/1706.03762}
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.values_fn = self.sinusoidal

    def _get_angles(self, pos, i, embed_d):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_d))
        return pos * angle_rates

    def _sin_cos(self, seq_len, embed_d):
        angle_rads = self._get_angles(
            np.arange(seq_len)[:, np.newaxis],
            np.arange(embed_d)[np.newaxis, :],
            embed_d,
        )

        # sin = even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # cos = odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def sinusoidal(self, seq_len: int, embed_d: int):
        """Create sinusoidal abosolute positional encodings

        Parameters
        ----------
        seq_len : int
            The length of the sequence
        embed_d : int
            The dimensionality of the embedding (how many are there?)

        Returns
        -------
        Tensor[float]
            positional encodings
        """
        # shape = (1, seq_len, embed_d)
        out = self._sin_cos(seq_len, embed_d)
        return out
