"""
Double-Buffered Pipelined Batched MCTS Benchmark
Compares:
- Standard Single-Batch MCTS (B = 64, 128, 256, 512, 1024)
- Double-Buffered Dual-Stream Pipelined MCTS (2x512 = 1024, 2x1024 = 2048)
Hardware: NVIDIA RTX 5060 Ti + 16 CPU cores
"""

import os
import sys
import time
from typing import List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import AlphaZeroResNet
from ml.dataset import UTTTEnv
from ml.batched_mcts import MCTSNode


class PipelinedBatchState:
    def __init__(self, envs: List[UTTTEnv], model: AlphaZeroResNet, device: str, c_puct: float = 2.0):
        self.envs = envs
        self.roots = [MCTSNode(env=env.clone()) for env in envs]
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.num_games = len(envs)

        # Initial root expansion
        batch_tensors = []
        batch_masks = []
        for root in self.roots:
            t, m = root.env.get_observation()
            batch_tensors.append(t)
            batch_masks.append(m)

        x_tensor = torch.from_numpy(np.array(batch_tensors, dtype=np.float32)).to(device)
        mask_tensor = torch.from_numpy(np.array(batch_masks, dtype=np.float32)).to(device)

        with torch.no_grad():
            with torch.amp.autocast("cuda"):
                root_logits, _ = model(x_tensor, mask_tensor)
            root_probs = torch.softmax(root_logits, dim=-1).cpu().numpy()

        for g_idx, root in enumerate(self.roots):
            legal = root.legal_moves
            if not legal:
                continue
            probs = root_probs[g_idx]
            for m in legal:
                next_env = root.env.clone()
                next_env.step(m)
                root.children[m] = MCTSNode(env=next_env, parent=root, parent_action=m, prior=float(probs[m]))
            root.is_expanded = True

        self.pending_leaves = []
        self.pending_x = None
        self.pending_mask = None
        self.pending_logits = None
        self.pending_values = None

    def select_and_prepare_leaves(self):
        """CPU Selection phase: selects leaf nodes and builds GPU input tensors."""
        leaves: List[MCTSNode] = []
        for root in self.roots:
            if root.is_terminal or not root.children:
                continue
            node = root
            while node.is_expanded and not node.is_terminal and node.children:
                _, next_node = node.select_child(c_puct=self.c_puct)
                node = next_node
            leaves.append(node)

        eval_leaves = []
        eval_tensors = []
        eval_masks = []
        for leaf in leaves:
            if not leaf.is_terminal and not leaf.is_expanded:
                t, m = leaf.env.get_observation()
                eval_leaves.append(leaf)
                eval_tensors.append(t)
                eval_masks.append(m)

        self.pending_leaves = eval_leaves
        self.all_leaves = leaves

        if eval_leaves:
            self.pending_x = torch.from_numpy(np.array(eval_tensors, dtype=np.float32)).to(self.device, non_blocking=True)
            self.pending_mask = torch.from_numpy(np.array(eval_masks, dtype=np.float32)).to(self.device, non_blocking=True)
        else:
            self.pending_x = None
            self.pending_mask = None

    def launch_gpu_eval_async(self, stream: torch.cuda.Stream):
        """Asynchronously dispatches GPU forward pass on given CUDA stream."""
        if self.pending_x is not None:
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    with torch.amp.autocast("cuda"):
                        self.pending_logits, self.pending_values = self.model(self.pending_x, self.pending_mask)

    def complete_and_backprop(self, stream: torch.cuda.Stream):
        """Waits for GPU completion on stream, expands leaves and backpropagates."""
        if self.pending_x is not None:
            stream.synchronize()
            leaf_probs = torch.softmax(self.pending_logits, dim=-1).cpu().numpy()
            leaf_vals = self.pending_values.squeeze(1).float().cpu().numpy()

            for i, leaf in enumerate(self.pending_leaves):
                legal = leaf.legal_moves
                probs = leaf_probs[i]
                for m in legal:
                    next_env = leaf.env.clone()
                    next_env.step(m)
                    leaf.children[m] = MCTSNode(
                        env=next_env, parent=leaf, parent_action=m, prior=float(probs[m])
                    )
                leaf.is_expanded = True
                leaf.total_value = float(leaf_vals[i])
                leaf.visit_count = 1

        for leaf in self.all_leaves:
            v = leaf.terminal_value if leaf.is_terminal else leaf.q_value
            curr = leaf.parent
            v = -v
            while curr is not None:
                curr.visit_count += 1
                curr.total_value += v
                v = -v
                curr = curr.parent


def run_pipelined_double_buffered_mcts(
    model: AlphaZeroResNet,
    batch_a_envs: List[UTTTEnv],
    batch_b_envs: List[UTTTEnv],
    num_simulations: int = 100,
    device: str = "cuda",
) -> Tuple[float, int]:
    """
    Executes Double-Buffered Pipelined MCTS across Batch A and Batch B simultaneously.
    While Batch A is evaluated on GPU Stream A, Batch B is selected/backpropagated on CPU!
    """
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    t0 = time.perf_counter()

    batch_a = PipelinedBatchState(batch_a_envs, model, device)
    batch_b = PipelinedBatchState(batch_b_envs, model, device)

    # Prime the pipeline: Select Batch A and dispatch to GPU
    batch_a.select_and_prepare_leaves()
    batch_a.launch_gpu_eval_async(stream_a)

    for _ in range(num_simulations):
        # 1. CPU processes Batch B selection while GPU computes Batch A
        batch_b.select_and_prepare_leaves()

        # 2. Complete Batch A (GPU done), expand & backprop on CPU
        batch_a.complete_and_backprop(stream_a)

        # 3. Launch Batch B on GPU Stream B
        batch_b.launch_gpu_eval_async(stream_b)

        # 4. CPU processes Batch A selection while GPU computes Batch B
        batch_a.select_and_prepare_leaves()

        # 5. Complete Batch B (GPU done), expand & backprop on CPU
        batch_b.complete_and_backprop(stream_b)

        # 6. Launch Batch A for next iteration
        batch_a.launch_gpu_eval_async(stream_a)

    # Drain pipeline
    stream_a.synchronize()
    batch_a.complete_and_backprop(stream_a)
    stream_b.synchronize()

    t1 = time.perf_counter()
    elapsed = t1 - t0
    total_evals = (len(batch_a_envs) + len(batch_b_envs)) * num_simulations

    return elapsed, total_evals


def benchmark_pipelined():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  DOUBLE-BUFFERED PIPELINED MCTS BENCHMARK ON NVIDIA RTX 5060 Ti")
    print("=" * 80)
    model = AlphaZeroResNet(num_channels=64, num_res_blocks=4).to(device)
    model.eval()

    test_configs = [
        ("Single Batch (Baseline)", 64, 0, 100),
        ("Single Batch", 128, 0, 100),
        ("Single Batch", 256, 0, 100),
        ("Single Batch", 512, 0, 100),
        ("Double-Buffered (2x256)", 256, 256, 100),
        ("Double-Buffered (2x512)", 512, 512, 100),
        ("Double-Buffered (2x1024)", 1024, 1024, 50),
    ]

    print(f"{'Configuration':<28} | {'Total Games':<12} | {'Sims':<6} | {'Time (s)':<10} | {'Throughput (evals/s)':<22}")
    print("-" * 80)

    for name, b_a, b_b, sims in test_configs:
        envs_a = [UTTTEnv() for _ in range(b_a)]
        envs_b = [UTTTEnv() for _ in range(b_b)] if b_b > 0 else []

        if b_b == 0:
            # Baseline single batch using standard MCTS
            from ml.batched_mcts import run_batched_gpu_mcts
            t0 = time.perf_counter()
            _ = run_batched_gpu_mcts(model, envs_a, num_simulations=sims, device=device)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            evals = b_a * sims
        else:
            elapsed, evals = run_pipelined_double_buffered_mcts(
                model=model,
                batch_a_envs=envs_a,
                batch_b_envs=envs_b,
                num_simulations=sims,
                device=device,
            )

        throughput = evals / max(elapsed, 0.001)
        total_games = b_a + b_b
        print(f"{name:<28} | {total_games:<12} | {sims:<6} | {elapsed:<10.3f} | {throughput:>20,.0f}")

    print("=" * 80)


if __name__ == "__main__":
    benchmark_pipelined()
