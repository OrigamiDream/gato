import tensorflow as tf

from gato.models.transformer import TransformerBlock
from gato.models.embedding import PatchPositionEncoding, ResidualEmbedding, LocalPositionEncoding, DiscreteEmbedding
from gato.models.tokenizers import ContinuousValueTokenizer

from tensorflow.keras import models
from gato import GatoConfig
from typing import Dict, Any, Union


class Gato(models.Model):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], trainable: bool = True, name: str = 'Gato', **kwargs):
        super(Gato, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.image_embedding = PatchEmbedding(config, trainable=trainable, name='ImagePatchEmbedding')
        self.discrete_embedding = DiscreteEmbedding(config, trainable=trainable, name='DiscreteEmbedding')
        self.continuous_encoding = ContinuousValueTokenizer(config, name='ContinuousValueEncoding')
        self.transformer = Transformer(config, trainable=trainable, name='Transformers')
        self.local_pos_encoding = LocalPositionEncoding(config, trainable=trainable, name='LocalPositionEncoding')

    def call(self, inputs, training=None, mask=None):
        # input_ids with (B, L, 768)
        # encoding with (B, L) or (B,)
        # row_pos and col_pos with tuple of (pos_from, pos_to)
        # obs_pos and obs_mask with (B, L) or (B,)
        input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask) = inputs
        # Encoding flags for embed masks
        # 0 - image
        # 1 - continuous
        # 2 - discrete (actions, texts)
        encoding = tf.one_hot(encoding, depth=3, dtype=tf.float32)

        ones = tf.ones((input_ids.shape[0], 1, self.config.layer_width), dtype=tf.float32)
        image_embed = self.image_embedding((input_ids, (row_pos, col_pos)), training=training)
        image_embed *= encoding[..., 0].transpose().matmul(ones)  # image patch masking

        # continuous value takes from first value of input_ids
        continuous_embed = self.continuous_encoding(input_ids[..., 0])
        continuous_embed = self.discrete_embedding(continuous_embed)
        continuous_embed *= encoding[..., 1].transpose().matmul(ones)  # continuous value masking

        discrete_embed = self.discrete_embedding(input_ids[..., 0])
        discrete_embed *= encoding[..., 2].transpose().matmul(ones)  # discrete value masking

        # Appendix C.3. Position Encodings > Local Observation Position Encodings
        # add local observation position encodings
        embed = image_embed + continuous_embed + discrete_embed
        embed += self.local_pos_encoding((obs_pos, obs_mask))

        hidden_states = self.transformer(embed)
        return hidden_states

    def get_config(self):
        return super(Gato, self).get_config()


class Transformer(models.Model):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 trainable: bool = True,
                 name: str = None,
                 **kwargs):
        super(Transformer, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.encoders = [TransformerBlock(config=self.config, trainable=trainable, name='EncoderBlock{}'.format(idx))
                         for idx in range(self.config.num_transformer_blocks)]

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)
        return x

    def get_config(self):
        return super(Transformer, self).get_config()


class PatchEmbedding(models.Model):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 trainable: bool = True,
                 name: str = None,
                 **kwargs):
        super(PatchEmbedding, self).__init__(trainable=trainable, name=name, **kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.residual_embedding = ResidualEmbedding(config, trainable=trainable, name='ResidualEmbedding')
        self.pos_encoding = PatchPositionEncoding(config, trainable=trainable, name='PatchPositionEncoding')

    def call(self, inputs, training=None, mask=None):
        input_ids, (row_pos, col_pos) = inputs
        patch_size = self.config.img_patch_size
        depth = self.config.input_dim // (patch_size * patch_size)

        x = input_ids.reshape((-1, input_ids.shape[1], patch_size, patch_size, depth))
        x = self.residual_embedding(x)
        x = self.pos_encoding((x, (row_pos, col_pos)))
        return x

    def get_config(self):
        return super(PatchEmbedding, self).get_config()
