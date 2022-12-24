from gato.models.transformer import TransformerBlock, PoolerOutput
from gato.models.embedding import PatchPositionEncoding, ResidualEmbedding, LocalPositionEncoding, DiscreteEmbedding
from gato.models.tokenizers import ContinuousValueTokenizer

from tensorflow.keras import layers, models
from gato import GatoConfig
from typing import Dict, Any, Union


class GatoTransformer(models.Model):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 trainable: bool = True,
                 name: str = 'gato_transformer'):
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        inputs = layers.Input(shape=(None, config.layer_width), name='inputs')
        x = inputs
        for idx in range(self.config.num_transformer_blocks):
            x = TransformerBlock(config=self.config,
                                 trainable=trainable,
                                 name='transformer_block_{}'.format(idx))(x)
        x = PoolerOutput(config, name='pooler_output')(x)
        super(GatoTransformer, self).__init__(inputs=inputs, outputs=x, trainable=trainable, name=name)

    def call(self, inputs, training=None, mask=None):
        return super(GatoTransformer, self).call(inputs, training, mask)

    def get_config(self):
        return super(GatoTransformer, self).get_config()


class PatchEmbedding(models.Model):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 input_shape=(64, 80, 3),
                 trainable: bool = True,
                 name: str = 'gato_patch_embedding'):
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        h, w = input_shape[0], input_shape[1]
        num_patches = (h // config.img_patch_size) * (w // config.img_patch_size)

        inputs = layers.Input(shape=input_shape, name='inputs')
        x = inputs
        x = ResidualEmbedding(config, trainable=trainable, name='residual_embedding')(x)
        x = layers.Conv2D(filters=config.layer_width,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding='valid',
                          name='embedding',
                          trainable=trainable)(x)
        x = layers.Reshape((num_patches, config.layer_width))(x)
        x = PatchPositionEncoding(embedding_dim=config.layer_width,
                                  img_height=input_shape[0], img_width=input_shape[1],
                                  config=config,
                                  trainable=trainable,
                                  name='patch_position_encoding')(x)
        super(PatchEmbedding, self).__init__(inputs=inputs, outputs=x, trainable=trainable, name=name)

    def call(self, inputs, training=None, mask=None):
        return super(PatchEmbedding, self).call(inputs, training, mask)

    def get_config(self):
        return super(PatchEmbedding, self).get_config()
