# Replay Buffer Routing Defect & 65% Stale Data Technical Challenges

The [`replay_buffer.py`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py) module manages a contiguous, pre-allocated 1,000,000-state stratified replay buffer used by [`champion_challenger.py`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py). This document analyzes the routing defect, data staleness accumulation, and gradient anchoring problems identified during our empirical diagnostic investigation.

---

## 1. The `tier="deep_mcts"` Routing Defect

### Problem
In [`champion_challenger.py`](file:///home/festivity/agy/uttt-codingame/ml/champion_challenger.py#L378), completed game samples are drained from the Rust Rayon worker and inserted into the replay buffer using:
```python
replay_buffer.add_samples_batch(f_drain, m_drain, p_drain, v_drain, tier="deep_mcts")
```

In [`replay_buffer.py`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L112), incoming samples are routed into one of three pre-allocated contiguous slices:
```python
is_deep_or_draw = (game_length >= 38) or (abs(val_f32) < 0.05) or (tier in ("deep_draw", "deep_mcts"))
```
Because the caller hardcodes `tier="deep_mcts"`, every single self-play sample evaluates `is_deep_or_draw = True`.

### Consequences:
1. **Unconditional Routing to `deep_draw`**: All samples are routed to `self.deep_draw_start` ($[300,000 \dots 649,999]$).
2. **Ring Eviction Lock**: Once `deep_draw_count` reached its capacity of $350,000$, ring eviction recycled exclusively within `deep_draw_ptr`:
   ```python
   slot = self.deep_draw_start + self.deep_draw_ptr
   self.deep_draw_ptr = (self.deep_draw_ptr + 1) % self.deep_draw_capacity
   ```
3. **Freezing of the `self_play` Slice**: The designated `self_play` partition ($[650,000 \dots 999,999]$) had its pointer permanently frozen at `sp_ptr = 0`. Over **6,671,591 cumulative lifetime insertions** (tracked in `meta[10]`), not a single sample was ever routed into or evicted from the `self_play` tier after its initial fill!
4. **Empirical Composition of the Trapped `deep_draw` Partition**:
   Inspection of `ml/replay_buffer_checkpoint.npz` (`meta = [1000000, 300000, 350000, 350000, 0, 300000, 71591, 350000, 0, 350000, 6671591]`) reveals that the active 350,000-state `deep_draw` partition contains:
   - **30,865 draw states** (`abs(value) < 0.05`, 8.8%)
   - **319,135 decisive self-play states** (`abs(value) >= 0.05`, 91.2%)
   Because all self-play was misrouted to `deep_draw`, 91.2% of the active deep_draw slice consists of fresh decisive self-play states trapped in the wrong partition.

---

## 2. Gradient Anchoring via 65% Stale Training Batches

### Problem
During Phase 2 (Challenger optimization), training batches of size 128 are sampled proportionally to tier counts ([`replay_buffer.py`](file:///home/festivity/agy/uttt-codingame/ml/replay_buffer.py#L193-L200)):
$$p_{\text{gm}} = \frac{N_{\text{gm}}}{N_{\text{total}}} = \frac{300,000}{1,000,000} = 30\%$$
$$p_{\text{deep}} = \frac{N_{\text{deep}}}{N_{\text{total}}} = \frac{350,000}{1,000,000} = 35\%$$
$$p_{\text{sp}} = \frac{N_{\text{sp}}}{N_{\text{total}}} = \frac{350,000}{1,000,000} = 35\%$$

Because:
- The Grandmaster tier ($300,000$ states) is static and never updated.
- The Self-Play tier ($350,000$ states) was frozen with legacy data from days earlier.

**Exactly 65% of every training batch ($30\% + 35\%$) consisted of completely static, legacy data!**

### Concrete Training Impact:
1. **Loss Stagnation**: In [`diagnostics.py`](file:///home/festivity/agy/uttt-codingame/ml/diagnostics.py), loss evaluated across the tiers showed:
   - Grandmaster tier loss: $2.6162$
   - Frozen Self-Play tier loss: $2.6123$
   - Active Deep Draw tier loss: $2.6453$
   The overall training loss hovered immutably at $\sim 2.62$ because $65\%$ of each batch was pulling the weights toward historical distributions.
2. **Policy Plasticity Loss**: When recent self-play discovered new winning refutations, the policy gradient was overwhelmed by historical gradients, severely slowing down reinforcement learning progress.

---

## 3. Disproportionate Grandmaster Sizing

### Problem
A 300,000-state static Grandmaster dataset was useful in Generation 0 to bootstrap basic move legality and opening principles. However, at **Generation 10 (Elo 2105.8)**, human Grandmaster play from CodinGame contains tactical sub-optimality that the self-play engine has already surpassed. Allocating $30\%$ of all training capacity to static human play actively discourages superhuman tactical discovery.

---

## 4. Invariants & Migration Requirements

1. **Zero Downtime Migration**: Checkpoint files (`replay_buffer_checkpoint.npz`) must remain binary-compatible or cleanly migratable without crashing active training loops.
2. **Contiguous Memory Safety**: Array slicing must remain contiguous (`np.ascontiguousarray`) to ensure zero-copy GPU transfers during batch sampling.
3. **Partition Normalization**: The sum of tier capacities must equal total buffer capacity ($N_{\text{gm}} + N_{\text{deep}} + N_{\text{sp}} = N_{\text{total}}$).
