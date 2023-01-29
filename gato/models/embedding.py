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
                 config: Union[GatoConfig, Dict[str, Any]],
                 trainable=True, name=None, *args, **kwargs):
        """
        Appendix C.3. Position Encodings
        """
        super(PatchPositionEncoding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        self.embedding_dim = self.config.layer_width
        self.discretize_depth = self.config.discretize_depth
        self.patch_size = self.config.img_patch_size

        self.row_embedding = layers.Embedding(self.discretize_depth, self.embedding_dim, name='row_embedding')
        self.col_embedding = layers.Embedding(self.discretize_depth, self.embedding_dim, name='col_embedding')

    def _discretize(self, pos):
        return tf.round(pos * self.discretize_depth)

    def _discretize_interval(self, interval):
        pos_from, pos_to = interval
        return self._discretize(pos_from), self._discretize(pos_to)

    def call(self, inputs, *args, **kwargs):
        # Appendix C.3. Position Encodings; Figure 15 | Patch position encodings.
        training = kwargs['training'] if 'training' in kwargs else False
        # input_ids must already be embedded by the resnet embedding function.
        # row_pos and col_pos must be intervals which is tuple of (pos_from, pos_to)
        # row_pos and col_pos must be normalized between [0, 1] to show their relativity.
        input_ids, (row_pos, col_pos) = inputs

        row_pos_from, row_pos_to = self._discretize_interval(row_pos)
        col_pos_from, col_pos_to = self._discretize_interval(col_pos)

        if training:
            # > During training a random index is uniformly sampled from the quantized interval.
            row_pos = row_pos_from + _randomized_positions(row_pos_from, row_pos_to)
            col_pos = col_pos_from + _randomized_positions(col_pos_from, col_pos_to)
        else:
            # > During evaluation we deterministically take the (rounded) mean of the interval.
            row_pos = _rounded_mean_positions(row_pos_from, row_pos_to)
            col_pos = _rounded_mean_positions(col_pos_from, col_pos_to)

        col_pos = tf.cast(col_pos, dtype=tf.int32)
        row_pos = tf.cast(row_pos, dtype=tf.int32)

        # > Once row and column position encoding are retrieved from the embedding table,
        # > they are added onto the token embedding produced by the resnet embedding function.
        return input_ids + self.row_embedding(row_pos) + self.col_embedding(col_pos)

    def get_config(self):
        config = super(PatchPositionEncoding, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config


class ResidualUnit(layers.Layer):

    def __init__(self, num_groups: int, filters: int, trainable=True, name=None, *args, **kwargs):
        super(ResidualUnit, self).__init__(trainable=trainable, name=name, *args, **kwargs)
        self.num_groups = num_groups
        self.filters = filters
        self.gn1 = self.gn2 = None
        self.conv1 = self.conv2 = None
        self.conv_proj = self.gn_proj = None

    def build(self, input_shape):
        self.gn1 = layers.GroupNormalization(groups=self.num_groups, name='gn1')
        self.gn2 = layers.GroupNormalization(groups=self.num_groups, name='gn2')
        self.conv1 = layers.Conv2D(filters=self.filters // 2, kernel_size=(3, 3), strides=(1, 1),
                                   use_bias=False, padding='same', name='conv1')
        self.conv2 = layers.Conv2D(filters=self.filters, kernel_size=(3, 3), strides=(2, 2),
                                   use_bias=False, padding='same', name='conv2')
        self.conv_proj = layers.Conv2D(filters=self.filters, kernel_size=(1, 1), strides=(2, 2),
                                       use_bias=False, padding='same', name='conv_proj')
        self.gn_proj = layers.GroupNormalization(groups=self.num_groups, name='gn_proj')

    def call(self, inputs, *args, **kwargs):
        # Supplementary Material B. Agent Data Tokenization Details; Figure 16
        # > This block uses the v2 ResNet architecture, GroupNorm (instead of LayerNorm) normalization,
        # > and GELU (instead RELU) activation functions.
        x = inputs

        residual = self.conv_proj(self.gn_proj(x))

        x = tf.nn.gelu(self.gn1(x))
        x = self.conv1(x)

        x = tf.nn.gelu(self.gn2(x))
        x = self.conv2(x)

        return x + residual


class ResidualEmbedding(layers.Layer):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], trainable=True, name=None, *args, **kwargs):
        """
        Appendix C.2. Embedding Function
        """
        super(ResidualEmbedding, self).__init__(trainable=trainable, name=name, *args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.root_conv = self.conv_proj = None
        self.residual_units = None
        self.num_patches = None

    def build(self, input_shape):
        if self.config.input_dim != self.config.layer_width:
            self.conv_proj = layers.Conv2D(filters=self.config.layer_width,
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding='same',
                                           use_bias=False,
                                           name='conv_proj')
        self.root_conv = models.Sequential([
            layers.Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2),
                          use_bias=False, padding='same', name='conv_root'),
            layers.GroupNormalization(groups=self.config.num_group_norm_groups, name='gn_root'),
            layers.Activation('gelu', name='act_root')
        ])
        self.residual_units = [ResidualUnit(num_groups=self.config.num_group_norm_groups,
                                            filters=96 * 2 ** (i + 1),
                                            name='residual_unit_{}'.format(i + 1))
                               for i in range(3)]

    def call(self, inputs, *args, **kwargs):
        # Section 2.1 Tokenization.
        x = self.root_conv(inputs)

        # NOTE: Page 3-4, Section 2.2 Embedding input tokens and setting output targets
        # > Tokens belonging to image patches for any time-step are embedded
        # > using a single ResNet block to obtain a vector per patch.

        # I don't think that transforming single 16x16 patch into feature map
        # with depth 768 at once does not give advantages coming from inductive bias.
        # This is currently discussing in issue #2
        for block in self.residual_units:
            x = block(x)
        if self.conv_proj is not None:
            x = self.conv_proj(x)
        x = tf.reshape(x, shape=(-1, inputs.shape[1], self.config.layer_width))
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
        self.embedding = layers.Embedding(self.config.token_sequence_length, self.config.layer_width)
        self.built = True

    def call(self, inputs, *args, **kwargs):
        # Appendix C.3. Position Encodings > Local Observation Position Encodings; Figure 18 | Local position encodings.
        # > Note that no position encodings are added to action tokens.

        # So I added `obs_mask` to mask the action token into zeros.
        obs_pos, obs_mask = inputs
        embed = self.embedding(obs_pos)

        ones = tf.ones((embed.shape[0], 1, self.config.layer_width), dtype=tf.float32)
        obs_mask = tf.cast(obs_mask, dtype=tf.float32)
        obs_mask = tf.matmul(obs_mask, ones, transpose_a=True)
        return embed * obs_mask

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
