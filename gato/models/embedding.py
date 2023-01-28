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
        input_shape = tf.TensorShape(input_shape)
        patch_size = self.config.img_patch_size
        h, w, c = input_shape[1:]
        width = patch_size * patch_size * c
        if width != self.config.layer_width:
            self.conv_proj = layers.Conv2D(filters=self.config.layer_width,
                                           kernel_size=(1, 1),
                                           strides=(1, 1),
                                           padding='same',
                                           use_bias=False,
                                           name='conv_proj')
        self.num_patches = (h // self.config.img_patch_size) * (w // self.config.img_patch_size)
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

    def create_patches(self, inputs):
        patch_size = self.config.img_patch_size
        x = tf.image.extract_patches(
            images=inputs,  # [B, H, W, C]
            sizes=(1, patch_size, patch_size, 1),
            strides=(1, patch_size, patch_size, 1),
            rates=(1, 1, 1, 1),
            padding='SAME'
        )
        x = tf.reshape(x, (-1, self.num_patches, patch_size, patch_size, inputs.shape[-1]))
        return tf.stop_gradient(x)

    def call(self, inputs, *args, **kwargs):
        # Section 2.1 Tokenization.
        # > Images are first transformed into sequences of non-overlapping
        # > 16 x 16 patches in raster order, as done in ViT (Dosovitskiy et al., 2020)
        x = self.create_patches(inputs)  # [B, patches, 16, 16, C]
        x = self.root_conv(x)

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
        x = tf.reshape(x, shape=(-1, self.num_patches, self.config.layer_width))
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
