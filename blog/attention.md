# Attention Is All You Need: A Formal Guide to Transformer Attention

*September 17, 2026 · Approx. 18-minute read*

## Abstract

[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) introduced the Transformer: a sequence model built from attention, feed-forward networks, residual connections, and normalization rather than recurrence. This note develops the encoder-style self-attention calculation from its matrix shapes upward, works through a numerical example, and collects the technical details that frequently matter in machine-learning interviews.

## 1. Notation and shapes

Let a sequence of $n$ token embeddings be stored row-wise in a matrix

$$
X = \begin{bmatrix} x_1^\top \\ x_2^\top \\ \vdots \\ x_n^\top \end{bmatrix}
\in \mathbb{R}^{n \times d_{\text{model}}}.
$$

For one attention head, learned projection matrices create a query, key, and value for every token:

$$
Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V,
$$

where $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. Therefore $Q,K \in \mathbb{R}^{n \times d_k}$ and $V \in \mathbb{R}^{n \times d_v}$.

| Quantity | Shape | Interpretation |
| --- | --- | --- |
| $QK^\top$ | $n \times n$ | Compatibility score for every query–key pair |
| $A$ | $n \times n$ | Row-normalized attention weights |
| $AV$ | $n \times d_v$ | One contextualized output vector per token |

The convention in this note is important: the $i$-th **row** of $A$ is the distribution used by token $i$ to read from all value vectors.

## 2. Scaled dot-product attention

The attention map and output are

$$
S = \frac{QK^\top}{\sqrt{d_k}} + M, \qquad
A = \operatorname{softmax}_{\text{row}}(S), \qquad
O = AV,
\tag{1}
$$

where $M \in \mathbb{R}^{n \times n}$ is an optional mask. The row-wise softmax is defined element by element as

$$
A_{ij} = \frac{\exp(S_{ij})}{\sum_{\ell=1}^{n}\exp(S_{i\ell})}.
\tag{2}
$$

Thus $A_{ij}$ measures how much token $i$ reads the value at token $j$. Every row of $A$ sums to one, so each output row is a convex combination:

$$
o_i = \sum_{j=1}^{n} A_{ij}v_j.
\tag{3}
$$

### Why divide by $\sqrt{d_k}$?

Suppose the coordinates of a query $q$ and key $k$ are independent, zero-mean, unit-variance random variables. Then

$$
q^\top k = \sum_{r=1}^{d_k}q_rk_r,
\qquad
\operatorname{Var}(q^\top k) \approx d_k.
$$

Without scaling, score magnitudes grow with $d_k$. The softmax then becomes too peaked, which produces near-zero gradients for most alternatives. Dividing by $\sqrt{d_k}$ keeps the score variance approximately constant:

$$
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right) \approx 1.
$$

This is a variance-control argument, not merely a numerical convention.

## 3. A matrix calculation by hand

Consider three two-dimensional token embeddings and, for clarity, choose identity projections: $W^Q=W^K=W^V=I_2$.

$$
X = \begin{bmatrix}
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix},
\qquad
Q=K=V=X.
$$

The unmasked scaled score matrix is

$$
S = \frac{QK^\top}{\sqrt{2}}
= \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 1 \\
1 & 1 & 2
\end{bmatrix}.
$$

For the first query, the score row is $[1,0,1]/\sqrt{2}$. Applying softmax gives, approximately,

$$
A_{1,:} = \operatorname{softmax}\left(
\begin{bmatrix}0.707 \\ 0 \\ 0.707\end{bmatrix}
\right)^\top
= \begin{bmatrix}0.401 & 0.198 & 0.401\end{bmatrix}.
$$

The first contextualized output is consequently

$$
o_1 = 0.401\begin{bmatrix}1 & 0\end{bmatrix}
+ 0.198\begin{bmatrix}0 & 1\end{bmatrix}
+ 0.401\begin{bmatrix}1 & 1\end{bmatrix}
= \begin{bmatrix}0.802 & 0.599\end{bmatrix}.
$$

The output is not a copy of one token. It is a content-dependent mixture of the available value vectors.

## 4. Causal masking and decoder attention

An encoder may attend to all input positions. An autoregressive decoder must not use future tokens while predicting the next token. For one-based positions, define the causal mask

$$
M_{ij} =
\begin{cases}
0, & j \leq i, \\
-\infty, & j > i.
\end{cases}
$$

For $n=4$, this is

$$
M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}.
$$

Because $\exp(-\infty)=0$, masked locations receive exactly zero probability after softmax. In practice, a very negative finite number is often used to avoid floating-point exceptions. Padding masks use the same mechanism to prevent attention to unused positions in a batch.

## 5. Multi-head attention

One attention head has a single learned similarity metric. Multi-head attention lets the model learn several metrics in parallel:

$$
\operatorname{head}_r = \operatorname{Attention}(XW_r^Q, XW_r^K, XW_r^V),
\qquad r=1,\ldots,h,
$$

$$
\operatorname{MultiHead}(X)
= \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O.
\tag{4}
$$

Typically, $d_k=d_v=d_{\text{model}}/h$, so concatenating all heads restores width $d_{\text{model}}$. If all projection matrices are square at model width, the $Q$, $K$, $V$, and output projections together have about $4d_{\text{model}}^2$ parameters. Changing $h$ repartitions the representation; it does not by itself multiply this dominant parameter count.

Multiple heads can specialize in different relationships—such as local syntax, long-range agreement, or entity reference—although these roles are learned rather than assigned by hand.

## 6. Position and the Transformer block

Attention is permutation-equivariant: if the rows of $X$ are shuffled, the rows of the output shuffle in the same way. Position information must therefore enter the model. The original paper adds sinusoidal positional encodings $P$ to token embeddings:

$$
P_{p,2i} = \sin\left(\frac{p}{10000^{2i/d_{\text{model}}}}\right),
\qquad
P_{p,2i+1} = \cos\left(\frac{p}{10000^{2i/d_{\text{model}}}}\right),
$$

and uses $X^{(0)}=E+P$. Modern models may instead use learned, relative, or rotary position methods; the fundamental requirement is to make order available to attention.

A common pre-layer-normalized Transformer block is

$$
\widetilde{H}^{(\ell)} = H^{(\ell)} + \operatorname{MHA}(\operatorname{LN}(H^{(\ell)})),
$$

$$
H^{(\ell+1)} = \widetilde{H}^{(\ell)} + \operatorname{FFN}(\operatorname{LN}(\widetilde{H}^{(\ell)})),
$$

with a position-wise feed-forward network such as

$$
\operatorname{FFN}(z) = W_2\,\operatorname{GELU}(W_1z+b_1)+b_2.
$$

Residual paths preserve an easy route for information and gradients; layer normalization stabilizes the scale of each token representation; the FFN supplies nonlinear feature transformation independently at each position.

## 7. Training objective, cost, and inference

For causal language modeling, the decoder is trained to predict each next token using only its prefix:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

With sequence length $n$ and model width $d$, a self-attention layer has roughly $\mathcal{O}(n^2d)$ compute from attention scores and value mixing, plus $\mathcal{O}(nd^2)$ projection compute. Its attention-weight memory is $\mathcal{O}(hn^2)$ for $h$ heads. In contrast, a recurrent layer has $\mathcal{O}(nd^2)$ compute and sequential dependence across positions.

During autoregressive generation, recomputing keys and values for the entire prefix would be wasteful. A **KV cache** stores past $K$ and $V$ tensors. Each new token still attends over the growing history, but the previous projections do not need to be recalculated.

## 8. From images to sequences: Vision Transformers

Transformers were adapted to computer vision in the Vision Transformer (ViT). Rather than replacing every CNN operation with attention directly, ViT first turns an image into a sequence of visual tokens.

For an image $I \in \mathbb{R}^{H \times W \times C}$ and a square patch size $P$, split the image into non-overlapping $P \times P$ patches. The number of patches is

$$
n = \frac{H}{P} \cdot \frac{W}{P},
$$

and each flattened patch has dimension $P^2C$. With a learned patch-embedding matrix $E \in \mathbb{R}^{(P^2C) \times d}$, the $i$-th visual token is

$$
z_i = \operatorname{vec}(I_i)E,
\qquad
z_i \in \mathbb{R}^{d}.
$$

ViT prepends a learned classification token $z_{\text{cls}}$ and adds a position embedding $P_{\text{pos}}$:

$$
Z^{(0)} =
\begin{bmatrix}
z_{\text{cls}} \\
z_1 \\
\vdots \\
z_n
\end{bmatrix}
 + P_{\text{pos}}
\in \mathbb{R}^{(n+1) \times d}.
\tag{5}
$$

This is the Transformer’s input sequence. After the final encoder layer, a classification head commonly reads the representation associated with $z_{\text{cls}}$.

### Two practical image-to-sequence routes

1. **Patchify the raw image (standard ViT).** The patch embedding above is equivalent to a convolution with kernel size $P$ and stride $P$, followed by flattening spatial locations into tokens.
2. **Use a CNN as a visual tokenizer (hybrid model).** If a CNN produces a feature map $F \in \mathbb{R}^{H' \times W' \times C'}$, reshape it into $H'W'$ vectors in $\mathbb{R}^{C'}$ and linearly project them to width $d$. Attention then mixes information across the CNN features.

CNNs build in locality and translation equivariance through small shared kernels. Full self-attention has no such local bias: any patch can attend to any other patch in one layer. This global receptive field is useful for long-range image relationships, but it usually makes ViTs more data-hungry unless they use large-scale pretraining, augmentation, or architectural priors.

### Attention cost for images

For $n$ image tokens, $h$ heads, and model width $d$, multi-head attention has:

$$
\text{projection cost} = \mathcal{O}(nd^2), \qquad
\text{attention mixing cost} = \mathcal{O}(n^2d),
$$

$$
\text{attention-score memory} = \mathcal{O}(hn^2).
$$

The quadratic term is why patch size matters. A $224 \times 224$ image with $P=16$ has $n=196$ patch tokens (or $197$ including the class token). Reducing the patch size to $P=8$ makes $n=784$, which is four times as many tokens and roughly sixteen times as many pairwise attention scores. Windowed and hierarchical architectures such as Swin Transformer reduce this cost by limiting attention to local windows and progressively merging tokens.

## 9. Technical interview questions

### 1. What are queries, keys, and values?

**Answer.** A query asks what the current token needs; keys describe what each source token offers; values contain the content that will be mixed. The score $q_i^\top k_j$ decides the weight placed on $v_j$. Keeping keys and values conceptually separate lets a model use one representation to match and another to transmit information.

### 2. Along which axis is softmax applied, and why?

**Answer.** In $A=\operatorname{softmax}_{\text{row}}(QK^\top/\sqrt{d_k})$, softmax is applied over keys $j$ for each fixed query $i$. Each row then sums to one and produces one weighted average of values. Applying it down columns would instead normalize how many queries select a key, which is not the standard attention operation.

### 3. Why is the attention matrix $n \times n$ even when $d_k$ is small?

**Answer.** $Q$ has shape $n\times d_k$ and $K^\top$ has shape $d_k\times n$, so their product compares every query position with every key position. The feature dimension $d_k$ is reduced in the dot product; the two sequence-position dimensions remain.

### 4. What is the difference between self-attention, cross-attention, and masked self-attention?

**Answer.** Self-attention forms $Q$, $K$, and $V$ from the same sequence. Cross-attention takes $Q$ from one sequence and $K,V$ from another, as in an encoder–decoder model. Masked self-attention is self-attention with a causal mask so position $i$ cannot access positions greater than $i$.

### 5. Does adding more heads always add more parameters?

**Answer.** Usually no. Holding $d_{\text{model}}$ fixed and setting each head width to $d_{\text{model}}/h$ keeps the combined $Q$, $K$, $V$, and output-projection sizes approximately unchanged. More heads change the factorization and may change optimization behavior, but they are not free: attention-score memory still grows linearly with $h$.

### 6. Why are residual connections and pre-layer normalization useful?

**Answer.** Residual paths allow a layer to make an incremental update rather than relearn an identity map. Pre-layer normalization presents well-scaled inputs to each sublayer and leaves an identity gradient path through the residual stream, which tends to improve stability in deep Transformers.

### 7. Why can attention be faster to train than an RNN but expensive for long contexts?

**Answer.** All attention scores for a layer can be computed with parallel matrix multiplications, while an RNN must process token states sequentially. However, dense attention compares every pair of positions, so score computation and memory grow quadratically in context length. Long-context methods reduce this cost by restricting, approximating, or reorganizing attention.

### 8. What does a KV cache change at inference time?

**Answer.** At step $t$, the new query is computed once and attends to cached keys and values for positions $1$ through $t$. This avoids recomputing old token projections. It reduces repeated work, but cache memory grows linearly with generated length, layer count, head count, and head dimension.

### 9. How does Transformer attention work, and what is its function?

**Answer.** Attention is a differentiable content-addressing operation. For every token $i$, the model compares its query $q_i$ against every key $k_j$, normalizes those compatibility scores into weights, and uses them to combine the corresponding values:

$$
o_i = \sum_{j=1}^{n}
\underbrace{
\frac{\exp(q_i^\top k_j/\sqrt{d_k})}
{\sum_{\ell=1}^{n}\exp(q_i^\top k_\ell/\sqrt{d_k})}
}_{\text{attention weight } A_{ij}}
v_j.
$$

The function of attention is to create a context-dependent representation: the same token can retrieve different information in different sentences, positions, or images. In self-attention, every token can exchange information with the rest of the sequence; in cross-attention, one sequence retrieves information from another. Unlike a fixed convolution kernel, the weights $A_{ij}$ are computed from the current input.

### 10. Why does scaled dot-product attention divide by $\sqrt{d_k}$?

**Answer.** If query and key coordinates have approximately zero mean and unit variance, the unscaled dot product has variance proportional to its dimension:

$$
\operatorname{Var}(q^\top k)
= \operatorname{Var}\left(\sum_{r=1}^{d_k}q_rk_r\right)
\approx d_k.
$$

As $d_k$ grows, unscaled logits become large. Softmax then assigns nearly all probability to one key, so most probabilities and gradients become very small. Scaling by $\sqrt{d_k}$ keeps logit variance near one:

$$
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right) \approx 1.
$$

This keeps the softmax in a trainable range. It does **not** force attention to be uniform; the learned projections can still make a relevant key dominant when the data support it.

### 11. How does ViT turn a CNN-style image into a sequence, and what are the cost trade-offs?

**Answer.** Standard ViT divides an $H \times W \times C$ image into $P \times P$ patches, flattens each patch to a vector in $\mathbb{R}^{P^2C}$, projects it to $\mathbb{R}^{d}$, adds positional information, and feeds the resulting $n=HW/P^2$ tokens into Transformer encoder blocks. A hybrid alternative first applies a CNN and treats each spatial location of its final feature map as a token.

The main trade-off is global context versus quadratic cost. Dense attention costs $\mathcal{O}(n^2d)$ time for score/value mixing and stores $\mathcal{O}(hn^2)$ attention weights during training. Smaller patches retain more visual detail but sharply increase $n$; larger patches are cheaper but can lose fine-grained information. CNNs are usually more efficient at high resolution because local convolution scales roughly linearly in the number of pixels for a fixed kernel.

### 12. What should you check when an attention implementation gives the wrong result?

**Answer.** First check shapes and transpose order: $QK^\top$ should end in $(n,n)$. Then confirm that softmax runs over the key axis, masking is added before softmax, masked scores use a sufficiently negative value, and batch/head dimensions are broadcast as intended. Finally, test a tiny matrix such as the worked example above; the rows of the unmasked attention matrix must sum to one.

## Further reading

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- The original Transformer paper’s [annotated architecture diagram](https://arxiv.org/pdf/1706.03762)
- Dosovitskiy et al. (2021), [*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)
