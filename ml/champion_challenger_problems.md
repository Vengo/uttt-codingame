# Champion Challenger Technical Challenges & Problem Domain

The [`champion_challenger.py`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py) module addresses several foundational algorithmic, statistical, and systems-level problems inherent in autonomous self-play reinforcement learning for Ultimate Tic-Tac-Toe.

---

## 1. Mitigation of First-Move and Positional Bias

### Problem
In Ultimate Tic-Tac-Toe, first-move initiative (Player 1 / X) and asymmetric opening subboard choices induce substantial win-rate disparities. Evaluating raw unanchored games leads to false promotions: a challenger can appear superior merely by winning a disproportionate share of Player 1 assignments.

### Implementation & Solution
- **Paired Mirrored Openings**: Each evaluated board position $P_k$ is instantiated twice:
  - Game $2k$: Challenger is Player 1 (X), Champion is Player 2 (O).
  - Game $2k+1$: Champion is Player 1 (X), Challenger is Player 2 (O).
- **Exact Cancellation**: If $P_k$ possesses an inherent positional advantage $V(P_k)$, both agents have an equal opportunity to exploit it. Win-rate calculations and pairwise split classifications (Challenger 2-0 double wins, Champion 0-2 double wins, and 1-1 symmetrical splits) isolate model skill from positional asymmetry.
- **Statistical Significance Guardrail**: Production runs enforce `eval_pairs >= 1024` ($2,048$ total games) and a strict $\ge 55\%$ promotion barrier, preventing noise-induced rating drift.

---

## 2. Exploration vs. Determinism Management

### Problem
During self-play data generation, high exploration is required to prevent policy collapse into degenerate sub-trees. Conversely, gating evaluation demands pure competitive exploitation. Furthermore, dropping search temperature to zero too early in self-play eliminates endgame tactical variety.

### Implementation & Solution
- **Two-Tier Self-Play Temperature Schedule**:
  - Plies $\le 20$: Temperature $T = 1.0$ (proportional visit sampling to explore diverse midgame paths).
  - Plies $> 20$: Temperature $T = 0.5$ (focused sampling on high-probability branches, maintaining non-zero entropy).
  - Temperature is never set to $0.0$ in self-play, ensuring endgame novelty and stochastic game endings.
- **Competitive Evaluation**: In gating matches ([`play_eval_match`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py#L116-L211) and [`run_parallel_mirror_tournament`](file:///home/festivity/agy/uttt-codingame/ml/parallel_mirror_eval.py)), Dirichlet exploration noise is set to $\epsilon = 0.0$ and actions are selected greedily ($T = 0.0$).

---

## 3. Multi-Architecture Compatibility

### Problem
The training pipeline supports both residual convolutional networks ([`AlphaZeroResNet`](file:///home/festivity/agy/uttt-codingame/ml/model.py#L35-L83)) and vision/board transformers ([`UTTTTransformer`](file:///home/festivity/agy/uttt-codingame/ml/transformer_model.py#L22-L139)). Loading checkpoints blindly risks state dict key mismatches and invalid graph initialization.

### Implementation & Solution
- In [`run_champion_challenger`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py#L267-L279), the checkpoint dictionary is inspected prior to model instantiation:
  ```python
  is_trans = any(k.startswith("transformer.") or k.startswith("sb_embed.") for k in sd.keys())
  ```
- If transformer keys are detected, [`UTTTTransformer`](file:///home/festivity/agy/uttt-codingame/ml/transformer_model.py#L22-L139) is loaded; otherwise, [`AlphaZeroResNet`](file:///home/festivity/agy/uttt-codingame/ml/model.py#L35-L83) is instantiated.
- **Conditional INT8 Export**: ResNet models are compiled and exported to the native Rust engine binary (`uttt/src/nn/model_int8.bin`) via [`export_model_int8`](file:///home/festivity/agy/uttt-codingame/ml/export_int8.py). Transformer models (which operate under FP16 AMP in PyTorch) bypass the ResNet-specific INT8 exporter, preserving baseline weights without corruption.

---

## 4. Asynchronous Atomic Telemetry

### Problem
External processes (such as frontend web monitors) continuously poll the telemetry JSON file (`training_status.json`). Standard non-atomic writes (`open("status.json", "w")`) can produce incomplete or corrupt JSON reads if read concurrently during write flushing.

### Implementation & Solution
- [`write_status`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py#L61-L80) serializes telemetry to a `.tmp` file and executes `os.replace`, guaranteeing POSIX filesystem atomic rename semantics.
- Status data contains phase indicators, hardware telemetry (GPU memory allocation, utilization percentage), GPU evaluation throughput, and generation history.

---

## 5. Invariants & Edge Cases

1. **Test-Mode Isolation**: When `--test-mode` is activated, production checkpoints (`champion.pt`, `replay_buffer_checkpoint.npz`, `champion_metadata.json`) are guarded against modification by remapping all targets to `.test_*` files.
2. **Buffer Drain Batching**: At the end of each generation cycle, samples are drained in bulk via `worker.drain_samples(max_count=100_000)` into the replay buffer. If $0$ samples are returned, insertion and buffer saving are safely skipped without error.
3. **Loss Stability**: Training applies mixed-precision autocasting (`torch.amp.autocast`), gradient unscaling, and gradient norm clipping (`max_norm=1.0`) to avoid gradient explosions during cross-entropy and MSE backpropagation.
