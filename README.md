# Unofficial Gato: A Generalist Agent

[[Deepmind Publication]](https://www.deepmind.com/publications/a-generalist-agent)
[[arXiv Paper]](https://arxiv.org/pdf/2205.06175.pdf)

This repository contains Deepmind's Gato architecture imitation in TensorFlow.

Since Deepmind only mentions parts of the architecture in its paper, We still don't know much about the model.<br>
However, I believe the paper is enough to imitate the architecture, I'm trying to do that with the open source community's help.

Currently, the repository supports the following operations:
- Transformer (via [`GatoTransformer`](https://github.com/OrigamiDream/gato/blob/main/gato/models/__init__.py#L10))
- Patch Position Encodings (via [`PatchPositionEmbedding`](https://github.com/OrigamiDream/gato/blob/main/gato/models/embedding.py#L11))
- Embedding Function (via [`ResidualEmbedding`](https://github.com/OrigamiDream/gato/blob/main/gato/models/embedding.py#L119))
- Local Observation Position Encodings (via [`LocalPositionEncoding`](https://github.com/OrigamiDream/gato/blob/main/gato/models/embedding.py#L209))
- Tokenizing Continuous Values (via [`ContinuousValueTokenizer`](https://github.com/OrigamiDream/gato/blob/main/gato/models/tokenizers.py#L30))

Action tokens are still a mystery in the paper, I need your help.

However, the repository lacks the following miscellaneous.
- Datasets (most important)
- Pre-trained tokenizers
- Training strategy

But, you can still explore the basic architecture of the Gato based on the paper.


## Paper Reviews

### Architecture Variants

> Appendix C.1. Transformer Hyperparameters

In the paper, Deepmind tested Gato with 3 architecture variants, `1.18B`, `364M`, and `79M`.<br>
I have named them as `large()`, `baseline()` and `small()` respectively in `GatoConfig`.

| Hyperparameters          | Large(1.18B) | Baseline(364M) | Small(79) |
|--------------------------|--------------|----------------|-----------|
| Transformer blocks       | 24           | 12             | 8         |
| Attention heads          | 16           | 12             | 24        |
| Layer width              | 2048         | 1536           | 768       |
| Feedforward hidden size  | 8192         | 6144           | 3072      |
| Key/value size           | 128          | 128            | 32        |


### Residual Embedding

> Appendix C.2. Embedding Function

There are no mentions that how many residual networks must be stacked for token embeddings.<br>
Therefore, I remain configurable in `GatoConfig`.

The blocks are consisted of:
- V2 ResNet architecture (ResNet1001)
- GroupNorm (instead of LayerNorm)
- GeLU (instead of ReLU)

Since the GroupNorm is not supported in TensorFlow, you need to install `tensorflow-addons`.

### Position Encodings

> Appendix C.3. Position Encodings

#### Patch Position Encodings

Like [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer) by Google, Gato takes the input images as raster-ordered 16x16 patches.<br>
Unlike the Vision Transformer model, however, Gato divides its patch encoding strategy into 2 phases, training and evaluation.

For high-performance computation in TensorFlow, I have used the following expressions.

$C$ and $R$ mean column and row-wise, and $F$ and $T$ mean `from` and `to` respectively.

$$
\begin{align}
  v^R_F &= \begin{bmatrix}
    0 & 32 & 64 & 96
  \end{bmatrix} \\
  v^R_T &= \begin{bmatrix}
    32 & 64 & 96 & 128
  \end{bmatrix} \\
  v^C_F &= \begin{bmatrix}
    0 & 26 & 51 & 77 & 102
  \end{bmatrix} \\
  v^C_T &= \begin{bmatrix}
    26 & 51 & 77 & 102 & 128
  \end{bmatrix} \\
  \\
  P_R &= \begin{cases}
    \mathsf{if} \ \mathsf{training} & v^R_F + \mathsf{uniform}(v^R_T - v^R_F) \\
    \mathsf{otherwise} & \mathsf{round}(\frac{v^R_F + v^R_T}{2})
  \end{cases} \\
  P_C &= \begin{cases}
    \mathsf{if} \ \mathsf{training} & v^C_F + \mathsf{uniform}(v^C_T - v^C_F) \\
    \mathsf{otherwise} & \mathsf{round}(\frac{v^C_F + v^C_T}{2})
  \end{cases} \\
  \\
  E^R_P &= P_R \cdot 1^{\mathsf{T}}_C \\
  E^C_P &= 1^{\mathsf{T}}_R \cdot P_C \\
  \\
  \therefore E &= E_I + E^R_P + E^C_P
\end{align}
$$

#### Local Observation Position Encodings

In the definition of Appendix B., text tokens, image patch tokens, and discrete & continuous values are observation tokens<br>
When Gato receives those values, they must be encoded with their own (local) time steps.

## Requirements

```bash
pip install tensorflow tensorflow-addons
```

## Contributing

This repository is still a work in progress.<br>
Currently, no downloads and no executables are provided.

I welcome many contributors who can help.

## License
Licensed under the [MIT license](https://github.com/OrigamiDream/gato/blob/main/LICENSE).