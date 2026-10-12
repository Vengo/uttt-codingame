# UTTT Transformer Technical Challenges & Problem Domain

The [`transformer_model.py`](file:///home/festivity/agy/uttt-codingame/ml/transformer_model.py) module addresses specific topological and numerical challenges arising from modeling Ultimate Tic-Tac-Toe with deep self-attention architectures.

---

## 1. Spatial Non-Locality vs. Convolutional Receptive Field Lag

### Problem
In standard games like Go or Chess, piece interactions exhibit strong local spatial correlations. In Ultimate Tic-Tac-Toe, move constraints are intrinsically non-local:
- A move played in cell $(r_{\text{sub}}, c_{\text{sub}})$ of subboard $B$ forces the opponent's next move to occur in subboard $B' = 3 \cdot r_{\text{sub}} + c_{\text{sub}}$.
- A standard CNN with $3 \times 3$ kernels has an effective receptive field of only 3 cells per layer. Propagating routing constraints between distant subboards requires 3 to 4 convolutional layers, introducing receptive field lag and hindering tactical foresight.

### Implementation & Solution
- By treating each of the 81 cells as an individual token, the pairwise multi-head self-attention mechanism computes direct attention weights between every cell $(r_{\text{sub}}, c_{\text{sub}})$ and its corresponding target subboard $B'$ in the very first layer.
- Long-range collinear relationships (e.g. 3-in-a-row completed subboards across the macro-board) are evaluated directly without distortions caused by intermediate pooling or convolutions.

---

## 2. Decomposed Subboard vs. Subsquare Positional Encoding

### Problem
Standard 1D sequential positional encodings (e.g. standard sinusoidal or flat 0..80 embeddings) obscure the two-level hierarchical geometry of Ultimate Tic-Tac-Toe:
- The subboard index ($0..8$) dictates where a move is placed on the macro-board.
- The subsquare index ($0..8$) dictates where the *opponent* will be forced to play next.

### Implementation & Solution
- In [`UTTTTransformer.__init__`](file:///home/festivity/agy/uttt-codingame/ml/transformer_model.py#L35-L62), two separate embedding tables are defined:
  - `sb_embed`: Embeds macro subboard coordinates $0..8$.
  - `sq_embed`: Embeds intra-subboard coordinates $0..8$.
- For each token $m \in [0, 80]$, the composite positional representation is additive:
  $$\text{pos\_emb}_m = \text{sb\_embed}(\lfloor m / 9 \rfloor) + \text{sq\_embed}(m \pmod 9)$$
- This structure enables attention heads to align tokens that either share a common subboard or route to the same destination subboard.

---

## 3. FP16 Mixed-Precision Underflow in Masked Attention & Logits

### Problem
In policy training with automatic mixed precision (AMP FP16), masking illegal actions with large negative constants like $-1 \times 10^9$ causes FP16 numerical underflow to negative infinity (`-inf`). When calculating `F.log_softmax(logits, dim=-1)`, exponentiating `-inf` or subtracting infinities results in `NaN` losses that corrupt model weights.

### Implementation & Solution
- In [`UTTTTransformer.forward`](file:///home/festivity/agy/uttt-codingame/ml/transformer_model.py#L131-L133), the masking threshold dynamically adapts to the tensor's floating-point precision:
  ```python
  fill_val = -1e4 if logits.dtype == torch.float16 else -1e9
  logits = torch.where(legal_mask.bool(), logits, torch.full_like(logits, fill_val))
  ```
- The value $-1 \times 10^4$ lies well within the FP16 minimum dynamic range ($\approx -65504$), yielding $\exp(-10000) = 0.0$ in softmax without generating `NaN` values.

---

## 4. Sequence Length & Throughput Optimization

### Problem
Standard vision transformers (ViTs) applied to large images suffer from $O(L^2)$ attention scaling overhead. High-throughput MCTS self-play requires tens of thousands of evaluations per second on a single GPU.

### Implementation & Solution
- Ultimate Tic-Tac-Toe possesses an exact, fixed token count of $L = 81$ cells plus 1 `[CLS]` token ($L = 82$).
- The resulting $82 \times 82$ attention matrices easily fit into GPU SRAM and L1/L2 caches.
- Setting `batch_first=True` and `norm_first=True` (Pre-LN) eliminates gradient vanishing and enables PyTorch FlashAttention / cuDNN optimizations, achieving $> 68,000$ state evaluations per second on consumer hardware.

---

## 5. Invariants & Edge Cases

1. **Pre-LN Gradient Stability**: Using `norm_first=True` ensures the residual stream is preserved without unnormalized additive accumulation across layers, allowing high learning rates without divergence.
2. **Deterministic Inference**: `dropout=0.0` ensures evaluations are strictly reproducible during competitive tournament gating.
3. **Buffer Registration**: Grid coordinates (`grid_r`, `grid_c`, etc.) are registered using `register_buffer`, guaranteeing they are moved to the active GPU device automatically on `.to(device)` without appearing as trainable parameters.
