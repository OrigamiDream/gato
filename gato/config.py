import copy

from typing import Dict, Any


class GatoConfig:

    @staticmethod
    def large():
        return GatoConfig(num_transformer_blocks=24,
                          num_attention_heads=16,
                          layer_width=2048,
                          feedforward_hidden_size=8192,
                          key_value_size=128)

    @staticmethod
    def baseline():
        return GatoConfig(num_transformer_blocks=12,
                          num_attention_heads=12,
                          layer_width=1536,
                          feedforward_hidden_size=6144,
                          key_value_size=128)

    @staticmethod
    def small():
        return GatoConfig(num_transformer_blocks=8,
                          num_attention_heads=24,
                          layer_width=768,
                          feedforward_hidden_size=3072,
                          key_value_size=32)

    def __init__(self, **kwargs):
        self.img_patch_size = kwargs.pop('img_patch_size', 16)

        # Section 2.3. Training
        self.token_sequence_length = kwargs.pop('token_sequence_length', 1024)

        # Section 2.1. Tokenization
        # Text - SentencePiece
        self.vocabulary_size = kwargs.pop('vocabulary_size', 32000)
        # Discrete values
        self.actions_size = kwargs.pop('actions_size', 1024)
        # Continuous values
        self.continuous_values_size = kwargs.pop('continuous_values_size', 1024)

        # Appendix C.1. Transformer Hyperparameters
        self.num_transformer_blocks = kwargs.pop('num_transformer_blocks', 8)
        self.num_attention_heads = kwargs.pop('num_attention_heads', 24)
        self.layer_width = kwargs.pop('layer_width', 768)
        self.feedforward_hidden_size = kwargs.pop('feedforward_hidden_size', 3072)
        self.key_value_size = kwargs.pop('key_value_size', 32)

        # Appendix E. Regularization
        self.dropout_rate = kwargs.pop('dropout_rate', 0.1)

        # Appendix C.2. Embedding Function
        self.num_group_norm_groups = kwargs.pop('num_group_norm_groups', 32)

        # Appendix C.3. Position Encodings > Patch Position Encodings
        self.discretize_depth = kwargs.pop('discretize_depth', 128)
        # Appendix C.3. Position Encodings > Local Observation Position Encodings
        self.local_position_encoding_size = kwargs.pop('local_position_encoding_size', 512)

    @property
    def embedding_input_size(self):
        return self.vocabulary_size + self.continuous_values_size + self.actions_size + 1

    @property
    def output_target_size(self):
        return self.vocabulary_size + self.actions_size

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        return output

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GatoConfig":
        config = cls(**config_dict)
        return config
