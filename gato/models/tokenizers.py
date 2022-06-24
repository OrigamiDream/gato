import tensorflow as tf

from gato import GatoConfig
from tensorflow.keras import layers, models
from typing import Union, Dict, Any


def mu_law_encode(x, mu=100, m=256):
    # Appendix B. Agent Data Tokenization Details
    sign = tf.math.sign(x)
    numerator = tf.math.log(tf.abs(x) * mu + 1.0)
    denominator = tf.math.log(m * mu + 1.0)
    return (numerator / denominator) * sign


def tokenize_continuous_values(x, mu=100, m=256, bins=1024, shift=None):
    # Appendix B. Agent Data Tokenization Details
    # > Finally, they are discretized using bins of uniform width on the domain [-1, 1].
    c = mu_law_encode(x, mu, m)

    # > We use 1024 bins and shift the resulting integers
    # > so they are not overlapping with the ones used for text tokens.
    c = (c + 1) * (bins / 2)
    c = tf.cast(c, tf.int32)
    if shift is not None:
        c += shift
    return c


class ContinuousValueTokenizer(models.Model):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 mu=100, m=256, bins=1024,
                 trainable=False, name='continuous_value_tokenizer'):
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        inputs = layers.Input(shape=(None,), name='inputs')
        outputs = tokenize_continuous_values(inputs, mu, m, bins, shift=config.vocabulary_size)

        super(ContinuousValueTokenizer, self).__init__(inputs=inputs, outputs=outputs, trainable=trainable, name=name)

    def call(self, inputs, training=None, mask=None):
        return super(ContinuousValueTokenizer, self).call(inputs, training, mask)

    def get_config(self):
        return super(ContinuousValueTokenizer, self).get_config()
