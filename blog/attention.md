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

The standard attention operation is

$$
\boxed{
\operatorname{Attention}(Q,K,V;M)
=
\operatorname{softmax}_{\text{row}}
\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
}
$$

Here $M$ is optional: setting $M=0$ recovers the usual unmasked formula. Writing the intermediate score, weight, and output matrices explicitly gives

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

Thus $A_{ij}$ measures how much token $i$ reads the value at token $j$. Before any attention-weight dropout, every row of $A$ sums to one, so each output row is a convex combination:

$$
o_i = \sum_{j=1}^{n} A_{ij}v_j.
\tag{3}
$$

### Why divide by $\sqrt{d_k}$?

Under the idealized assumption that all query and key coordinates are mutually independent, zero-mean, unit-variance random variables,

$$
q^\top k = \sum_{r=1}^{d_k}q_rk_r,
\qquad
\mathbb{E}[q_rk_r]=0,
\qquad
\operatorname{Var}(q_rk_r)=1.
$$

Consequently, independence gives

$$
\operatorname{Var}(q^\top k)
= \sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r)
= d_k,
\qquad
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right)=1.
$$

Without scaling, score *differences* can grow with $d_k$, making softmax too peaked and leaving very small gradients for most alternatives. Softmax is invariant to a shared shift, $\operatorname{softmax}(s+c\mathbf 1)=\operatorname{softmax}(s)$, so large scores alone are not the issue; differences between scores are. For learned queries and keys the independence and unit-variance assumptions need not hold, making this a useful variance-control heuristic rather than a guarantee.

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

Computing softmax for every row gives the complete attention matrix

$$
A \approx
\begin{bmatrix}
0.401112 & 0.197776 & 0.401112 \\
0.197776 & 0.401112 & 0.401112 \\
0.248255 & 0.248255 & 0.503490
\end{bmatrix}.
$$

The complete contextualized output is

$$
O=AV \approx
\begin{bmatrix}
0.802224 & 0.598888 \\
0.598888 & 0.802224 \\
0.751745 & 0.751745
\end{bmatrix}.
$$

For example, the first contextualized output is

$$
o_1 = 0.401\begin{bmatrix}1 & 0\end{bmatrix}
+ 0.198\begin{bmatrix}0 & 1\end{bmatrix}
+ 0.401\begin{bmatrix}1 & 1\end{bmatrix}
= \begin{bmatrix}0.802 & 0.599\end{bmatrix}.
$$

The output is not a copy of one token. It is a content-dependent mixture of the available value vectors.

## 4. Attention masks and causal decoder attention

The optional matrix $M$ constrains which query--key pairs are allowed. In full self-attention it has shape $n\times n$; in cross-attention it can have shape $n_q\times n_k$. Padding masks, local-window masks, and causal masks are all examples. An encoder may attend to all input positions, while an autoregressive decoder must not use future tokens when predicting the next token. For one-based positions, the causal mask is

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

Because $\exp(-\infty)=0$, a masked location receives exactly zero probability whenever its row has at least one allowed finite score. For example,

$$
\operatorname{softmax}([2,-\infty,1])
=
\left[
\frac{e^2}{e^2+e},\;0,\;\frac{e}{e^2+e}
\right].
$$

Implementations may instead use Boolean masks or sufficiently negative, dtype-compatible finite values. A finite sentinel can yield a tiny nonzero weight before numerical underflow; it does not by itself make a fully masked row correct. Mathematically, $\operatorname{softmax}([ -\infty,-\infty,-\infty])$ is undefined, so fully masked rows require explicit handling under the chosen implementation contract.

For next-token training, the causal condition $j\leq i$ is consistent with shifted labels:

```text
Inputs:   [BOS, x1, x2, x3]
Targets:  [x1,  x2, x3, x4]
```

At position $i$, the model may read the current shifted input and its prefix, but it cannot read a future target token.

## 5. Multi-head attention

One attention head has one learned compatibility function. Multi-head attention lets the model learn several such functions in parallel:

$$
\operatorname{head}_r = \operatorname{Attention}(XW_r^Q, XW_r^K, XW_r^V),
\qquad r=1,\ldots,h,
$$

$$
\operatorname{MultiHead}(X)
= \operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O.
\tag{4}
$$

Let $d=d_{\text{model}}$ and use the usual head width $d_h=d/h$. Then each head has rectangular projection matrices

$$
W_r^Q,W_r^K,W_r^V \in \mathbb{R}^{d\times d_h}.
$$

Concatenating all $h$ heads restores width $d$. Equivalently, the combined query, key, and value projections are each $d\times d$, and the output projection is $d\times d$. Excluding biases, the total is therefore

$$
3h\left(d\frac{d}{h}\right)+d^2=4d^2
$$

projection weights. Changing $h$ repartitions the representation; it does not by itself multiply this dominant parameter count. The score $q_i^\top k_j=x_i^\top W^Q(W^K)^\top x_j$ is a learned compatibility score, not necessarily a symmetric similarity metric.

Multiple heads can specialize in different relationships—such as local syntax, long-range agreement, or entity reference—although these roles are learned rather than assigned by hand.

## 6. Position and the Transformer block

Unmasked self-attention without positional signals is permutation-equivariant. More precisely, define

$$
F_M(X)=
\operatorname{softmax}_{\mathrm{row}}
\left(
\frac{XW^Q(XW^K)^\top}{\sqrt{d_k}}+M
\right)XW^V.
$$

For any permutation matrix $\Pi$,

$$
F_{\Pi M\Pi^\top}(\Pi X)=\Pi F_M(X).
$$

Thus $F_0(\Pi X)=\Pi F_0(X)$: unmasked attention alone cannot distinguish sequence order. Positional encodings, position-dependent biases, and structured masks can introduce order. In particular, a fixed causal mask already breaks arbitrary permutation equivariance. The original paper adds sinusoidal positional encodings $P$ to token embeddings:

$$
P_{p,2i} = \sin\left(\frac{p}{10000^{2i/d_{\text{model}}}}\right),
\qquad
P_{p,2i+1} = \cos\left(\frac{p}{10000^{2i/d_{\text{model}}}}\right),
$$

If $E$ denotes unscaled embeddings, the original input is $X^{(0)}=\sqrt{d_{\text{model}}}E+P$ (before dropout). Writing $E+P$ is equivalent when that scale is absorbed into the embedding definition. Modern models may instead use learned, relative, or rotary position methods.

The following equations show a common pre-layer-normalized Transformer block with a GELU feed-forward network. The original 2017 Transformer instead used post-layer normalization and ReLU:

$$
\widetilde{H}^{(\ell)} = H^{(\ell)} + \operatorname{MHA}(\operatorname{LN}(H^{(\ell)})),
$$

$$
H^{(\ell+1)} = \widetilde{H}^{(\ell)} + \operatorname{FFN}(\operatorname{LN}(\widetilde{H}^{(\ell)})),
$$

With the row-wise convention used here, its position-wise feed-forward network is

$$
\operatorname{FFN}(Z)=\operatorname{GELU}(ZW_1+b_1)W_2+b_2,
$$

where $Z\in\mathbb{R}^{n\times d}$, $W_1\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$, and $W_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$; biases are broadcast across rows.

Residual paths preserve an easy route for information and gradients; layer normalization stabilizes the scale of each token representation; the FFN supplies nonlinear feature transformation independently at each position.

## 7. Training objective, cost, and inference

For causal language modeling, the decoder is trained to predict each next token using only its prefix:

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

With sequence length $n$ and model width $d$, a self-attention layer has roughly $\mathcal{O}(n^2d)$ arithmetic from attention scores and value mixing, plus $\mathcal{O}(nd^2)$ projection arithmetic. A conventional implementation that explicitly materializes all attention scores or weights uses $\mathcal{O}(hn^2)$ elements for $h$ heads. That full quadratic storage is not unavoidable: memory-efficient exact attention implementations can avoid materializing the complete $n\times n$ matrix in high-bandwidth memory, while dense attention arithmetic remains quadratic. In contrast, a recurrent layer has $\mathcal{O}(nd^2)$ compute and sequential dependence across positions.

During autoregressive generation, recomputing keys and values for the entire prefix would be wasteful. A **KV cache** stores past $K$ and $V$ tensors. Each new token still attends over the growing history, but the previous projections do not need to be recalculated. Assuming equal key and value head dimensions, a conventional cache stores approximately

$$
2BLT\,h_{\mathrm{KV}}d_{\mathrm{head}}
$$

scalar elements for batch size $B$, layers $L$, cached length $T$, and $h_{\mathrm{KV}}$ key/value heads. In ordinary multi-head attention, $h_{\mathrm{KV}}=h$ and $d_{\mathrm{head}}=d/h$, giving $2BLTd$ elements: increasing $h$ at fixed model width does not itself enlarge the cache. Grouped-query attention reduces storage by using fewer key/value heads.

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
\text{explicit attention-score or weight storage} = \mathcal{O}(hn^2).
$$

The quadratic term is why patch size matters. A $224 \times 224$ image with $P=16$ has $n=196$ patch tokens (or $197$ including the class token). Reducing the patch size to $P=8$ makes $n=784$, which is four times as many tokens and roughly sixteen times as many pairwise attention scores. This concerns attention pairs, not necessarily a sixteenfold increase in end-to-end runtime. Windowed and hierarchical architectures such as Swin Transformer reduce this cost by limiting attention to local windows and progressively merging tokens.

## 9. Technical interview questions

### 1. What are queries, keys, and values?

**Answer.** A query asks what the current token needs; keys describe what each source token offers; values contain the content that will be mixed. The score $q_i^\top k_j$ decides the weight placed on $v_j$. Keeping keys and values conceptually separate lets a model use one representation to match and another to transmit information.

### 2. Along which axis is softmax applied, and why?

**Answer.** In $A=\operatorname{softmax}_{\text{row}}(QK^\top/\sqrt{d_k})$, softmax is applied over keys $j$ for each fixed query $i$. Each row then sums to one and produces one weighted average of values. Applying it down columns would instead normalize how many queries select a key, which is not the standard attention operation.

### 3. Why is the attention matrix $n \times n$ even when $d_k$ is small?

**Answer.** $Q$ has shape $n\times d_k$ and $K^\top$ has shape $d_k\times n$, so their product compares every query position with every key position. The feature dimension $d_k$ is reduced in the dot product; the two sequence-position dimensions remain.

### 4. What is the difference between self-attention, cross-attention, and masked self-attention?

**Answer.** Self-attention forms $Q$, $K$, and $V$ from the same sequence. Cross-attention takes $Q$ from one sequence and $K,V$ from another, as in an encoder–decoder model. Masked self-attention restricts which position pairs may interact. Causal self-attention is the special case that blocks future positions; padding and local-window masks are other examples.

### 5. Does adding more heads always add more parameters?

**Answer.** Usually no. Holding $d_{\text{model}}$ fixed and setting each head width to $d_{\text{model}}/h$ keeps the combined $Q$, $K$, $V$, and output-projection sizes approximately unchanged. More heads change the factorization and may change optimization behavior. If attention weights are explicitly materialized, their storage grows linearly with $h$; fused implementations need not retain that complete matrix.

### 6. Why are residual connections and pre-layer normalization useful?

**Answer.** Residual paths allow a layer to make an incremental update rather than relearn an identity map. Pre-layer normalization presents well-scaled inputs to each sublayer and leaves an identity gradient path through the residual stream, which tends to improve stability in deep Transformers.

### 7. Why can attention be faster to train than an RNN but expensive for long contexts?

**Answer.** All attention scores for a layer can be computed with parallel matrix multiplications, while an RNN must process token states sequentially. However, dense attention compares every pair of positions, so its arithmetic grows quadratically in context length. Conventional implementations also have quadratic score/weight storage, although memory-efficient exact kernels can avoid retaining the full matrix. Long-context methods reduce cost by restricting, approximating, or reorganizing attention.

### 8. What does a KV cache change at inference time?

**Answer.** At step $t$, the new query is computed once and attends to cached keys and values for positions $1$ through $t$. This avoids recomputing old token projections. Cache storage grows linearly with generated length, batch size, layer count, number of KV heads, and head dimension: approximately $2BLT h_{\mathrm{KV}}d_{\mathrm{head}}$ scalar elements. For ordinary multi-head attention at fixed $d_{\text{model}}$, $h_{\mathrm{KV}}d_{\mathrm{head}}=d_{\text{model}}$, so adding query heads alone does not increase this count; grouped-query attention reduces it by sharing KV heads.

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

**Answer.** Under the idealized assumption that query and key coordinates are mutually independent, zero-mean, and unit-variance, the unscaled dot product has variance proportional to its dimension:

$$
\operatorname{Var}(q^\top k)
= \operatorname{Var}\left(\sum_{r=1}^{d_k}q_rk_r\right)
= d_k.
$$

As $d_k$ grows, unscaled logit differences can grow. Softmax is unchanged by a shared shift, but large differences can assign nearly all probability to one key, leaving most probabilities and gradients very small. Scaling by $\sqrt{d_k}$ keeps the idealized logit variance at one:

$$
\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right) = 1.
$$

For learned projections, correlation and changing variance make this an approximation rather than a guarantee. It does **not** force attention to be uniform; the learned projections can still make a relevant key dominant when the data support it.

### 11. How does ViT turn a CNN-style image into a sequence, and what are the cost trade-offs?

**Answer.** Standard ViT divides an $H \times W \times C$ image into $P \times P$ patches, flattens each patch to a vector in $\mathbb{R}^{P^2C}$, projects it to $\mathbb{R}^{d}$, adds positional information, and feeds the resulting $n=HW/P^2$ tokens into Transformer encoder blocks. A hybrid alternative first applies a CNN and treats each spatial location of its final feature map as a token.

The main trade-off is global context versus quadratic cost. Dense attention costs $\mathcal{O}(n^2d)$ arithmetic for score/value mixing; conventional training implementations may explicitly store $\mathcal{O}(hn^2)$ attention weights, while memory-efficient exact kernels can avoid that full storage. Smaller patches retain more visual detail but sharply increase $n$; larger patches are cheaper but can lose fine-grained information. CNNs are usually more efficient at high resolution because local convolution scales roughly linearly in the number of pixels for a fixed kernel.

### 12. What should you check when an attention implementation gives the wrong result?

**Answer.** First check shapes and transpose order: full self-attention has $QK^\top\in\mathbb{R}^{n\times n}$, while cross-attention has shape $n_q\times n_k$ and cached decoding often has a $1\times T$ score row. Then confirm that softmax runs over the key axis, masking is added before softmax, every query has at least one allowed key, and batch/head dimensions broadcast as intended. Finally, test a tiny matrix such as the worked example above. Before attention-weight dropout, each unmasked softmax row must sum to one.

## Further reading

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- The original Transformer paper’s [annotated architecture diagram](https://arxiv.org/pdf/1706.03762)
- Dosovitskiy et al. (2021), [*An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*](https://arxiv.org/abs/2010.11929)
- Dao et al. (2022), [*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*](https://arxiv.org/abs/2205.14135)
- Ainslie et al. (2023), [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245)
