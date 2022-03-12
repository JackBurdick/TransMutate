import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
Include positional information
"""


class IncludePositional(Layer):
    """Base class for specified positional inclusion functions

    shape: (batch, len, dim)
    """

    def __init__(self, match_len=True, match_d=False, **kwargs):
        super().__init__(**kwargs)

        self.match_len = match_len
        self.match_d = match_d

        self.include_fn = self._include_fn

    def _include_fn(self, sequence, positional):
        raise NotImplementedError(
            "Must implement a `score_fn(sequence, positional)` in the subclass"
        )

    def call(self, sequence, positional):
        if self.match_len:
            # slice positional len to the length of sequence (batch, len, dim)
            positional = tf.slice(
                positional,
                [0, 0, 0],
                [
                    tf.shape(positional)[0],
                    tf.shape(sequence)[1],
                    tf.shape(positional)[2],
                ],
            )

        if self.match_d:
            # slice positional d to the dim of the sequence (batch, len, dim)
            assert tf.shape(positional)[1] >= tf.shape(sequence)[1], (
                f"cannot match dim, positional d ({tf.shape(positional)[1]})"
                f" is larger than sequence d ({tf.shape(sequence)[1]})"
                f"\n > sequence ({tf.shape(sequence)})"
                f"\n > positional ({tf.shape(positional)})"
            )
            positional = tf.slice(
                positional,
                [0, 0, 0],
                [
                    tf.shape(positional)[0],
                    tf.shape(positional)[1],
                    tf.shape(sequence)[2],
                ],
            )

        assert tf.shape(sequence)[1] == tf.shape(positional)[1], (
            f"sequence len (shape[1]): {tf.shape(sequence)[1]}"
            f" != positional len (shape[1]) {tf.shape(positional)[1]}"
        )

        out = self.include_fn(sequence=sequence, positional=positional)

        return out


class Additive(IncludePositional):
    """Add positional embedding to the sequence

    By default the positional embedding will be sliced to match both the
    sequence length and dim

    e.g.
    shape: (batch, seq_len, d)
    pos: [1,  110, 32]
    seq: [16, 100, 16]
    > (pos will be sliced to [1, 100, 16]) [both l and d sliced to match]
    out: [16, 100, 48]

    NOTE: if positional is (1, 100, 5) and sequence is (16, 100, 5) the
    positional seq will broadcast such that pos + seq works as expected i.e.
    each seq is added by pos
    """

    def __init__(self, match_len=True, match_d=True, **kwargs):
        super().__init__(match_len, match_d, **kwargs)
        self.include_fn = self.additive_positional

    def additive_positional(self, sequence, positional):
        out = sequence + positional
        return out


class Concatenation(IncludePositional):
    """Concatenate positional embedding on dim of sequence

    By default the positional embedding will be sliced to match the length, but
    the d dim will be unaltered.
    e.g.
    shape: (batch, seq_len, d)
    pos: [1,  110, 32]
    seq: [16, 100, 16]
    > (pos will be sliced to [1, 100, 32]), but d will be unaffected
    out: [16, 100, 48]
    """

    def __init__(self, axis=-1, match_len=True, match_d=False, **kwargs):
        super().__init__(match_len, match_d, **kwargs)
        self.include_fn = self.concat_positional
        self.axis = axis

    def concat_positional(self, sequence, positional):
        out = tf.concat([sequence, positional], axis=self.axis)
        return out