# AlphaZero ResNet Technical Challenges & Problem Domain

The [`model.py`](file:///home/festivity/agy/uttt-codingame/ml/model.py) module addresses spatial encoding representation, coordinate transformations, and mixed-precision numerical edge cases for convolutional neural network architectures in Ultimate Tic-Tac-Toe.

---

## 1. Spatial Channel Encoding & Hierarchical State Representation

### Problem
Ultimate Tic-Tac-Toe possesses a hierarchical state space: 9 micro-boards nested inside 1 macro-board. A raw board view containing only player cell marks ($X$ and $O$) forces the convolutional network to implicitly compute:
1. Which subboards are already won or tied.
2. Which subboard is the forced legal target for the active move.
Without explicit spatial channels, convolutions struggle to detect macro-board wins or distinguish between free-choice states and constrained single-subboard turns.

### Implementation & Solution
- In [`encode_board`](file:///home/festivity/agy/uttt-codingame/ml/model.py#L102-L183), five explicit 2D spatial feature channels are constructed:
  - Channels 0 & 1 capture raw local stone placements relative to the active player.
  - Channels 2 & 3 broadcast macro-board status (subboards won by active vs. opponent player) across all 9 cells of each subboard, providing immediate spatial awareness to $3 \times 3$ filters.
  - Channel 4 broadcasts the valid subboard constraint, identifying whether the player is restricted to a single subboard or enjoys a free move across all incomplete subboards.
- Full completion detection accounts for both won subboards and tied subboards (`xbb[sb] | obb[sb] == 0x1FF`), correctly identifying open boards when a move directs the opponent to an already-full board.

---

## 2. Bijective Non-Contiguous Spatial Coordinate Mapping

### Problem
Flat move indices $m \in [0, 80]$ are sequentially indexed by subboard:
$$m = 9 \cdot \text{subboard} + \text{subsquare}$$
However, on the physical 2D board, subboard 0 occupies rows $0..2$, cols $0..2$; subboard 1 occupies rows $0..2$, cols $3..5$; subboard 3 occupies rows $3..5$, cols $0..2$. Flat index adjacency does not match physical spatial adjacency (e.g., cell 8 in subboard 0 is physically adjacent to cell 6 in subboard 1, but separated by 7 in flat indices).

### Implementation & Solution
- [`move_index_to_row_col`](file:///home/festivity/agy/uttt-codingame/ml/model.py#L86-L93) and [`row_col_to_move_index`](file:///home/festivity/agy/uttt-codingame/ml/model.py#L95-L100) establish a strictly bijective spatial mapping:
  ```python
  subboard = move_index // 9
  subsquare = move_index % 9
  row = (subboard // 3) * 3 + (subsquare // 3)
  col = (subboard % 3) * 3 + (subsquare % 3)
  ```
- This mapping ensures that $3 \times 3$ convolutional filters operate over physically contiguous spatial neighborhoods while preserving standard 81-dimensional move indexing for policy outputs.

---

## 3. Dynamic Masking Stability Under FP16 Autocasting

### Problem
In reinforcement learning self-play, illegal moves must be assigned near-zero probabilities after softmax. A common pattern is masking illegal logits with $-1 \times 10^9$. However, under PyTorch Automatic Mixed Precision (`torch.amp.autocast`), tensors are evaluated in IEEE 754 half-precision (`float16`). The minimum representable normal number in FP16 is $\approx -65504$. Passing $-1 \times 10^9$ causes FP16 overflow to `-inf`. Exponentiating `-inf` or subtracting logits during backpropagation frequently results in `NaN` gradient spikes.

### Implementation & Solution
- In [`AlphaZeroResNet.forward`](file:///home/festivity/agy/uttt-codingame/ml/model.py#L73-L75), the masking constant dynamically checks the tensor's runtime data type:
  ```python
  if legal_mask is not None:
      fill_val = -1e4 if p.dtype == torch.float16 else -1e9
      p = torch.where(legal_mask.bool(), p, torch.full_like(p, fill_val))
  ```
- The constant $-1 \times 10^4$ ensures that illegal move probabilities are infinitesimal ($\approx 0.0$) without triggering FP16 overflow or numerical instability.

---

## 4. Invariants & Edge Cases

1. **Tanh Bounded Range**: The value head terminates in `torch.tanh(self.fc_v2(v))`, guaranteeing strict scalar bounds in $[-1.0, 1.0]$ matching game outcome values.
2. **Batch Normalization Tracking**: All convolutions in the trunk and heads omit bias terms (`bias=False`), as biases are subsumed by adjacent `BatchNorm2d` affine shift parameters, preventing redundant parameters during INT8 quantization.
3. **Free Move Subboard Masking**: If `last_super_index >= 9` or if the directed subboard is already full/won, `target_subboard_mask` correctly activates all non-completed subboards (`(~complete_bbs) & 0x1FF`).
