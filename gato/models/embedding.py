import math

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import layers, regularizers, activations
from gato import GatoConfig
from typing import Dict, Any, Union


def _randomized_positions(from_v, to_v):
    pos = tf.random.uniform(from_v.shape, minval=0, maxval=1, dtype=tf.float32)
    pos = pos * tf.cast(to_v - from_v, dtype=tf.float32)
    pos = tf.cast(pos, dtype=tf.int32)
    return pos


def _rounded_mean_positions(from_v, to_v):
    pos = tf.cast(from_v + to_v, tf.float32)
    pos = pos / 2
    pos = tf.round(pos)
    pos = tf.cast(pos, dtype=tf.int32)
    return pos


def _broadcast(row_pos, col_pos, row_ones, col_ones):
    # broadcast (5,) to (20,) with column-axis
    row_pos = tf.expand_dims(row_pos, 1)
    row_pos = tf.matmul(row_pos, col_ones, transpose_b=True)
    row_pos = tf.reshape(row_pos, (-1,))
    row_pos = tf.stop_gradient(row_pos)

    # broadcast (4,) to (20,) with row-axis
    col_pos = tf.expand_dims(col_pos, 1)
    col_pos = tf.matmul(row_ones, col_pos, transpose_b=True)
    col_pos = tf.reshape(col_pos, (-1,))
    col_pos = tf.stop_gradient(col_pos)

    return row_pos, col_pos


class PatchPositionEncoding(layers.Layer):

    def __init__(self,
                 embedding_dim, img_height, img_width,
                 config: Union[GatoConfig, Dict[str, Any]],
                 trainable=True, name=None, *args, **kwargs):
        """
        Appendix C.3. Position Encodings
        """
        super(PatchPositionEncoding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        assert img_height % self.config.img_patch_size == 0, 'Height must be divided by patch size with no remainders'
        assert img_width % self.config.img_patch_size == 0, 'Width must be divided by patch size with no remainders'

        self.embedding_dim = embedding_dim
        self.discretize_depth = self.config.discretize_depth
        self.height = img_height
        self.width = img_width
        self.patch_size = self.config.img_patch_size

        self.rows_from = self.rows_to = self.cols_from = self.cols_to = None
        self.row_embedding = self.col_embedding = None
        self.row_train_pos = self.col_train_pos = None
        self.row_eval_pos = self.col_eval_pos = None

    def _discretize(self, pos):
        return round(pos * self.discretize_depth)

    def _discretize_interval(self, axis_num):
        axis_from = []
        axis_to = []
        for index in range(axis_num // self.patch_size):
            from_pos = index * self.patch_size / axis_num
            to_pos = (index + 1) * self.patch_size / axis_num
            axis_from.append(self._discretize(from_pos))
            axis_to.append(self._discretize(to_pos))
        return axis_from, axis_to

    def build(self, input_shape):
        # Appendix C.3. Position Encodings; Figure 15 | Patch position encodings.
        rows_from, rows_to = self._discretize_interval(self.height)
        cols_from, cols_to = self._discretize_interval(self.width)

        self.rows_from = tf.convert_to_tensor(rows_from, dtype=tf.int32)
        self.rows_to = tf.convert_to_tensor(rows_to, dtype=tf.int32)
        self.cols_from = tf.convert_to_tensor(cols_from, dtype=tf.int32)
        self.cols_to = tf.convert_to_tensor(cols_to, dtype=tf.int32)

        self.row_embedding = layers.Embedding(self.discretize_depth, self.embedding_dim, name='row_embedding')
        self.col_embedding = layers.Embedding(self.discretize_depth, self.embedding_dim, name='col_embedding')

        row_ones = tf.ones(shape=(self.height // self.patch_size, 1), dtype=tf.int32)
        col_ones = tf.ones(shape=(self.width // self.patch_size, 1), dtype=tf.int32)

        # > During training a random index is uniformly sampled from the quantized interval.
        self.row_train_pos, self.col_train_pos = _broadcast(self.rows_from + _randomized_positions(self.rows_from, self.rows_to),
                                                            self.cols_from + _randomized_positions(self.cols_from, self.cols_to),
                                                            row_ones, col_ones)
        # > During evaluation we deterministically take the (rounded) mean of the interval.
        self.row_eval_pos, self.col_eval_pos = _broadcast(_rounded_mean_positions(self.rows_from, self.rows_to),
                                                          _rounded_mean_positions(self.cols_from, self.cols_to),
                                                          row_ones, col_ones)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        # Appendix C.3. Position Encodings
        training = kwargs['training'] if 'training' in kwargs else False
        row_pos, col_pos = (
            (self.row_train_pos, self.col_train_pos) if training else (self.row_eval_pos, self.col_eval_pos)
        )
        # > Once row and column position encoding are retrieved from the embedding table,
        # > they are added onto the token embedding produced by the resnet embedding function.
        return inputs + self.row_embedding(row_pos) + self.col_embedding(col_pos)

    def get_config(self):
        config = super(PatchPositionEncoding, self).get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'config': self.config.to_dict(),
        })
        return config


class ResidualEmbedding(layers.Layer):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], trainable=True, name=None, *args, **kwargs):
        """
        Appendix C.2. Embedding Function
        """
        super(ResidualEmbedding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.num_blocks = 0

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        h, w = input_shape[1], input_shape[2]

        num_h_patches = h // self.config.img_patch_size
        num_w_patches = w // self.config.img_patch_size

        num_h_pools = int(math.log2(h // num_h_patches))
        num_w_pools = int(math.log2(w // num_w_patches))

        assert num_h_pools == num_w_pools, (
            'Number of the downsampling blocks must be same (H:{} != W:{})'.format(num_h_pools, num_w_pools)
        )
        self.num_blocks = min(num_h_pools, num_w_pools) + 1

        self.built = True

    def call(self, inputs, *args, **kwargs):
        def _conv_block(num_filters, kernel_size, strides=(1, 1), name=None):
            return layers.Conv2D(filters=num_filters,
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding='same',
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=regularizers.L2(l2=1e-4),
                                 use_bias=False,
                                 name=name)

        def _norm_activation_block(conv, index, block_id):
            conv = tfa.layers.GroupNormalization(groups=32,
                                                 name='conv{}_block{}_preact_gn'.format(index, block_id))(conv)
            conv = layers.Lambda(lambda v: activations.gelu(v),
                                 name='conv{}_block{}_preact_gelu'.format(index, block_id))(conv)
            return conv

        x = _conv_block(32, kernel_size=(3, 3), strides=(1, 1), name='conv0_conv')(inputs)
        for conv_id in range(self.num_blocks):
            conv_id += 1
            filters = conv_id * 128  # from 128
            stride_size = (1, 1) if conv_id == 1 else (2, 2)

            # Appendix C.2. Embedding Function
            x = _norm_activation_block(x, conv_id, 1)
            residual = _conv_block(filters,
                                   kernel_size=(1, 1),
                                   strides=stride_size,
                                   name='conv{}_residual'.format(conv_id))(x)
            # Block 1
            x = _conv_block(filters // 4,
                            kernel_size=(1, 1),
                            name='conv{}_block1_conv'.format(conv_id))(x)

            # Block 2
            x = _norm_activation_block(x, conv_id, 2)
            x = _conv_block(filters // 4,
                            kernel_size=(3, 3),
                            strides=stride_size,
                            name='conv{}_block2_conv'.format(conv_id))(x)

            # Block 3
            x = _norm_activation_block(x, conv_id, 3)
            x = _conv_block(filters,
                            kernel_size=(1, 1),
                            name='conv{}_block3_conv'.format(conv_id))(x)

            # Residual connection
            x = x + residual
        return x

    def get_config(self):
        config = super(ResidualEmbedding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class LocalPositionEncoding(layers.Layer):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], trainable=True, name=None, *args, **kwargs):
        """
        Appendix C.3. Position Encodings > Local Observation Position Encodings
        """
        super(LocalPositionEncoding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.embedding = None

    def build(self, input_shape):
        self.embedding = layers.Embedding(self.config.local_position_encoding_size, self.config.layer_width)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        input_size = inputs.shape[1]
        pos = tf.range(input_size, dtype=tf.int32)
        return inputs + self.embedding(pos)

    def get_config(self):
        config = super(LocalPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config
