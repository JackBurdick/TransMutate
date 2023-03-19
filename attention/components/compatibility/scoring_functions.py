import tensorflow as tf
from tensorflow.keras.layers import Layer

"""
# TODO: is this q=source and target=k correct?
target ~ key
source ~ query
"""


class ScoringFunction(Layer):
    """Base class for specified scoring functions"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_fn = self._score_fn

    def _score_fn(self, target, source):
        raise NotImplementedError(
            "Must implement a `score_fn(target, source)` in the subclass"
        )

    def call(self, target, source):
        score = self.score_fn(target=target, source=source)
        return score


class DotProduct(ScoringFunction):
    """Dot-Product (Luong) Attention

    @inproceedings{Luong2015EffectiveAT,
        title={Effective Approaches to Attention-based
               Neural Machine Translation},
        author={Thang Luong and Hieu Pham and Christopher D. Manning},
        booktitle={EMNLP},
        year={2015}
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_fn = self.dot_product

    def dot_product(self, target, source):
        scores = tf.matmul(target, source, transpose_b=True)
        return scores


class ScaledDotProduct(DotProduct):
    """Scaled Dot-Product Attention

    @article{Vaswani2017AttentionIA,
        title={Attention is All you Need},
        author={Ashish Vaswani and Noam M. Shazeer and
                Niki Parmar and Jakob Uszkoreit and
                Llion Jones and Aidan N. Gomez and
                Lukasz Kaiser and Illia Polosukhin},
        journal={ArXiv},
        year={2017},
        volume={abs/1706.03762}
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_fn = self.scaled_dot_product

    def scaled_dot_product(self, target, source):
        # standard dot product
        matmul_ts = self.dot_product(target, source)
        # scaling factor
        n = tf.cast(tf.shape(source)[-1], tf.float32)
        scores = matmul_ts / tf.math.sqrt(n)
        return scores


class _Additive(ScoringFunction):
    """Additive (Bahdanau) Attention


    NOTE: I'm not convinced the following is implemented correctly but have
    included with a leading _

    It is based on a combination of:
    - https://medium.com/analytics-vidhya/neural-machine-translation-
        using-bahdanau-attention-mechanism-d496c9be30c3
    - https://github.com/tensorflow/addons/blob/
        8cec33fcaaf1cf90aec7bdd55a0fcdbb251ce5c2/
        tensorflow_addons/seq2seq/attention_wrapper.py
    - https://machinelearningmastery.com/the-bahdanau-attention-mechanism/

    @inproceedings{Luong2015EffectiveAT,
        title={Effective Approaches to Attention-based
               Neural Machine Translation},
        author={Thang Luong and Hieu Pham and Christopher D. Manning},
        booktitle={EMNLP}, year={2015}
    }
    """

    def __init__(
        self,
        units=None,
        kernel_initializer_str="glorot_uniform",
        dtype=tf.float32,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.kernel_initializer_str = kernel_initializer_str
        self.units = units

        if self.units is None:
            raise ValueError("please specify depth of the query mechanism")
        assert isinstance(self.units, int), TypeError(
            f"units ({units}) should be type {int}, not {type(units)}"
        )
        self.w1 = tf.keras.layers.Dense(units, name="w1", use_bias=False, dtype=dtype)
        self.w2 = tf.keras.layers.Dense(units, name="w2", use_bias=False, dtype=dtype)
        self.score_fn = self.additive

    def build(self, input_shape):
        super().build(input_shape=input_shape)
        self.attention_vals = self.add_weight(
            "attention_vals",
            [self.units],
            dtype=self.dtype,
            initializer=self.kernel_initializer_str,
        )
        self.built = True

    def additive(self, target, source):
        # setup
        target = self.w2(target)
        source = self.w1(source)

        # [batch, ...] -> [batch, 1, ...] (for broadcast)
        source = tf.expand_dims(source, 1)

        # calculate scores
        additive = tf.tanh(source + target)
        o = self.attention_vals * additive
        scores = tf.reduce_sum(o, axis=-1)
        return scores
