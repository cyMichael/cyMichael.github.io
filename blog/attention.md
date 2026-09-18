# Attention Is All You Need
*September 17, 2026 • Reading Time: 8 mins*

The paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) revolutionized natural language processing by introducing the **Transformer** architecture. Prior to this, sequence modeling relied heavily on Recurrent Neural Networks (RNNs) and Long Short-Term Memory networks (LSTMs). The Transformer discarded recurrence entirely in favor of an architecture solely based on attention mechanisms.

## 1. Scaled Dot-Product Attention

At the core of the Transformer is the Scaled Dot-Product Attention. The input consists of queries ($Q$), keys ($K$), and values ($V$). The attention function maps a query and a set of key-value pairs to an output.

```tex
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

Here, $d_k$ is the dimension of the keys. The dot products of the query with all keys are computed, scaled by $\sqrt{d_k}$, and passed through a softmax function to obtain the weights on the values.

### Interview Key Point: Why divide by $\sqrt{d_k}$?

**Answer:** Without scaling, for large values of $d_k$, the dot products grow very large in magnitude. This pushes the softmax function into regions where it has extremely small gradients (the vanishing gradient problem). Dividing by $\sqrt{d_k}$ helps keep the variance of the dot product roughly at 1 (assuming $Q$ and $K$ have zero mean and unit variance), ensuring stable gradients during training.

## 2. Multi-Head Attention

Instead of performing a single attention function, the authors found it beneficial to linearly project the queries, keys, and values $h$ times with different, learned linear projections.

```tex
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
```

```tex
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

### Interview Key Point: What is the advantage of Multi-Head Attention?

**Answer:** Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. It basically gives the model multiple "perspectives" on the sequence.

## 3. Positional Encoding

Since the Transformer contains no recurrence and no convolution, it has no inherent notion of sequence order. To inject some information about the relative or absolute position of the tokens in the sequence, **Positional Encodings** are added to the input embeddings.

```tex
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
```
```tex
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
```

### Interview Key Point: Why use sinusoidal positional encodings?

**Answer:** The authors hypothesized that sinusoidal encodings would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$. Furthermore, they can extrapolate to sequence lengths longer than the ones encountered during training.

## 4. Complexity and Parallelization

A major advantage of Transformers is computational efficiency and parallelizability compared to RNNs.

- **Self-Attention Layer:** Complexity per layer is $\mathcal{O}(n^2 \cdot d)$, where $n$ is sequence length and $d$ is representation dimension.
- **Recurrent Layer:** Complexity per layer is $\mathcal{O}(n \cdot d^2)$.

### Interview Key Point: When is a Transformer computationally worse than an RNN?

**Answer:** When the sequence length $n$ is significantly larger than the representation dimension $d$. Because self-attention scales quadratically with sequence length $\mathcal{O}(n^2)$, it becomes a bottleneck for very long sequences (which is why models like Longformer or Linformer were later developed). However, for typical lengths, $\mathcal{O}(n^2 \cdot d)$ is highly parallelizable across tokens, whereas RNNs are strictly sequential, making Transformers fundamentally faster to train on modern GPUs.
