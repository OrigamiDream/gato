import math

import tensorflow as tf

from tensorflow.keras import layers, models
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
        self.downsampling_convolution = None

    def block_v2(self, x, filters, kernel_size=3, stride=1, conv_residual=False, name=None):
        # Appendix C.2. Embedding Function
        preact = layers.GroupNormalization(groups=self.config.num_group_norm_groups,
                                           axis=-1,
                                           epsilon=1.001e-5,
                                           name='{}_preact_gn'.format(name))(x)
        preact = layers.Activation('gelu', name='{}_preact_gelu'.format(name))(preact)

        if conv_residual:
            residual = layers.Conv2D(4 * filters, 1, strides=stride, name='{}_residual_conv'.format(name))(preact)
        else:
            residual = layers.MaxPooling2D(1, strides=stride,
                                           name='{}_residual_pool'.format(name))(x) if stride > 1 else x

        x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name='{}_0_conv'.format(name))(preact)

        # Block 1
        x = layers.GroupNormalization(groups=self.config.num_group_norm_groups,
                                      axis=-1,
                                      epsilon=1.001e-5,
                                      name='{}_1_gn'.format(name))(x)
        x = layers.Activation('gelu', name='{}_1_gelu'.format(name))(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='{}_1_pad'.format(name))(x)
        x = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name='{}_1_conv'.format(name))(x)

        # Block 2
        x = layers.GroupNormalization(groups=self.config.num_group_norm_groups,
                                      axis=-1,
                                      epsilon=1.001e-5,
                                      name='{}_2_gn'.format(name))(x)
        x = layers.Activation('gelu', name='{}_2_gelu'.format(name))(x)
        x = layers.Conv2D(4 * filters, 1, name='{}_2_conv'.format(name))(x)

        x = layers.Add(name=name + '_add')([residual, x])
        return x

    def stack_v2(self, x, filters, num_blocks, stride=2, name=None):
        x = self.block_v2(x, filters, conv_residual=True, name='{}_block1'.format(name))
        for i in range(2, num_blocks):
            x = self.block_v2(x, filters, name='{}_block{}'.format(name, i))
        x = self.block_v2(x, filters, stride=stride, name='{}_block{}'.format(name, num_blocks))
        return x

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
        inputs = layers.Input(shape=(h, w, input_shape[-1]), name='downsampling_inputs')

        # ResNet50V2 (BatchNorm → GroupNorm, ReLU → GeLU)
        x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(inputs)
        x = layers.Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
        x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
        x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

        x = self.stack_v2(x, 64, 3, name='conv2')
        x = self.stack_v2(x, 128, 4, name='conv3')
        x = self.stack_v2(x, 256, 6, name='conv4')
        x = self.stack_v2(x, 512, 3, stride=1, name='conv5')

        model = models.Model(inputs=inputs, outputs=x, name='downsampling_convolution')

        current_shape = (h, w)
        current_layer_name = None
        for layer in model.layers:
            if 'preact_gn' not in layer.name:
                continue

            current_h, current_w = layer.output.shape[1:3]
            is_desired_shape = current_shape[0] == num_h_patches and current_shape[1] == num_w_patches
            is_smaller_shape = current_h < num_h_patches and current_w < num_w_patches
            if is_desired_shape and is_smaller_shape:
                break

            current_layer_name = layer.name
            current_shape = (current_h, current_w)

        self.downsampling_convolution = models.Model(inputs=inputs,
                                                     outputs=model.get_layer(name=current_layer_name).output,
                                                     name='downsampling_convolution')
        self.built = True

    def call(self, inputs, *args, **kwargs):
        return self.downsampling_convolution(inputs)

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
        self.embedding = layers.Embedding(self.config.token_sequence_length, self.config.layer_width)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        input_size = tf.shape(inputs)[1]
        pos = tf.range(start=0, limit=input_size, dtype=tf.int32)
        return inputs + self.embedding(pos)

    def get_config(self):
        config = super(LocalPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config


class DiscreteEmbedding(layers.Layer):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], trainable=True, name=None, *args, **kwargs):
        super(DiscreteEmbedding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.embedding = None

    def build(self, input_shape):
        # Appendix C.1. Transformer Hyperparameters
        # Shared Embedding
        with tf.name_scope('discrete_shared_embedding'):
            self.embedding = layers.Embedding(self.config.embedding_input_size,
                                              self.config.layer_width,
                                              name='discrete_embedding')
        self.built = True

    def call(self, inputs, *args, **kwargs):
        return self.embedding(inputs)

    def get_config(self):
        config = super(DiscreteEmbedding, self).get_config()
        config.update({
            'config': self.config.to_dict()
        })
        return config
