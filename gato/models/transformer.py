import tensorflow as tf

from tensorflow.keras import layers, models, activations
from gato import GatoConfig
from typing import Dict, Any, Union


class TransformerBlock(layers.Layer):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 trainable: bool = True,
                 name: str = None,
                 *args, **kwargs):
        super(TransformerBlock, self).__init__(trainable, name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.attention = self.feed_forward = self.dropout = None
        self.layer_norm1 = self.layer_norm2 = None

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        hidden_size = input_shape[-1]

        self.attention = layers.MultiHeadAttention(num_heads=self.config.num_attention_heads,
                                                   key_dim=self.config.key_value_size,
                                                   value_dim=self.config.key_value_size,
                                                   dropout=self.config.dropout_rate,
                                                   name='attention')
        self.dropout = layers.Dropout(self.config.dropout_rate, name='attention_dropout')
        self.feed_forward = models.Sequential(layers=[
            layers.Dense(units=self.config.feedforward_hidden_size,
                         activation='linear',
                         name='dense_intermediate'),
            # Appendix.C.1. Transformer Hyperparameters
            # Activation Function: GEGLU
            layers.Lambda(lambda x: activations.gelu(x, approximate=False), name='gelu'),
            layers.Dropout(self.config.dropout_rate, name='dropout_intermediate'),
            layers.Dense(units=hidden_size,
                         activation='linear',
                         name='dense'),
            layers.Dropout(self.config.dropout_rate, name='dropout'),
        ], name='feed_forward')
        self.layer_norm1 = layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
        self.layer_norm2 = layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')

    def call(self, inputs, *args, **kwargs):
        # Appendix.C.1. Transformer Hyperparameters
        # Layer Normalization: Pre-Norm
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.attention(x, x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(inputs)
        x = self.feed_forward(x)
        x = x + residual
        return x

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config
