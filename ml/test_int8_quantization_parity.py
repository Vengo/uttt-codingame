"""
INT8 Quantization Parity Test
Compares PyTorch floating-point predictions (FP32/FP16) against INT8 quantized inference
across diverse game positions to verify policy fidelity and value calibration.
"""

import os
import sys
import struct
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import AlphaZeroResNet
from ml.replay_buffer import ContiguousStratifiedReplayBuffer


class Int8ModelEmulator:
    """Python reference emulator matching uttt/src/nn/mod.rs exact fixed-point INT8 evaluation."""

    def __init__(self, bin_path: str):
        with open(bin_path, "rb") as f:
            bytes_data = f.read()

        magic = bytes_data[:8]
        if magic != b"UTTTNN01":
            raise ValueError(f"Invalid magic: {magic}")

        self.num_channels, self.num_res_blocks = struct.unpack("<II", bytes_data[8:16])
        offset = 16

        def read_conv3x3(in_c, out_c):
            nonlocal offset
            w_len = out_c * 3 * 3 * in_c
            w = np.frombuffer(bytes_data[offset : offset + w_len], dtype=np.int8).reshape(out_c, 3, 3, in_c)
            offset += w_len
            b_len = out_c * 4
            b = np.frombuffer(bytes_data[offset : offset + b_len], dtype=np.int32)
            offset += b_len
            mult = struct.unpack("<f", bytes_data[offset : offset + 4])[0]
            offset += 4
            return w, b, mult

        def read_conv1x1(in_c, out_c):
            nonlocal offset
            w_len = out_c * in_c
            w = np.frombuffer(bytes_data[offset : offset + w_len], dtype=np.int8).reshape(out_c, in_c)
            offset += w_len
            b_len = out_c * 4
            b = np.frombuffer(bytes_data[offset : offset + b_len], dtype=np.int32)
            offset += b_len
            mult = struct.unpack("<f", bytes_data[offset : offset + 4])[0]
            offset += 4
            return w, b, mult

        def read_linear(in_f, out_f):
            nonlocal offset
            w_len = out_f * in_f
            w = np.frombuffer(bytes_data[offset : offset + w_len], dtype=np.int8).reshape(out_f, in_f)
            offset += w_len
            b_len = out_f * 4
            b = np.frombuffer(bytes_data[offset : offset + b_len], dtype=np.int32)
            offset += b_len
            mult = struct.unpack("<f", bytes_data[offset : offset + 4])[0]
            offset += 4
            return w, b, mult

        self.conv0_w, self.conv0_b, self.conv0_m = read_conv3x3(5, self.num_channels)
        self.res_blocks = []
        for _ in range(self.num_res_blocks):
            w1, b1, m1 = read_conv3x3(self.num_channels, self.num_channels)
            w2, b2, m2 = read_conv3x3(self.num_channels, self.num_channels)
            self.res_blocks.append((w1, b1, m1, w2, b2, m2))

        self.conv_p_w, self.conv_p_b, self.conv_p_m = read_conv1x1(self.num_channels, 2)
        self.fc_p_w, self.fc_p_b, self.fc_p_m = read_linear(2 * 81, 81)

        self.conv_v_w, self.conv_v_b, self.conv_v_m = read_conv1x1(self.num_channels, 1)
        self.fc_v1_w, self.fc_v1_b, self.fc_v1_m = read_linear(81, 64)
        self.fc_v2_w, self.fc_v2_b, self.fc_v2_m = read_linear(64, 1)

    def forward(self, input_hwc_i8: np.ndarray, legal_mask_bool: np.ndarray):
        # input_hwc_i8: (9, 9, 5)
        # 1. conv0
        padded = np.zeros((11, 11, 5), dtype=np.int32)
        padded[1:10, 1:10, :] = input_hwc_i8.astype(np.int32)
        c = self.num_channels
        out_c0 = np.zeros((9, 9, c), dtype=np.int8)

        for r in range(9):
            for col in range(9):
                patch = padded[r : r + 3, col : col + 3, :]  # (3, 3, 5)
                # acc = bias + sum(patch * w)
                acc = self.conv0_b + np.sum(patch * self.conv0_w, axis=(1, 2, 3))
                scaled = np.round(acc * self.conv0_m).astype(np.int32)
                out_c0[r, col] = np.clip(scaled, 0, 127).astype(np.int8)

        act = out_c0
        # 2. ResBlocks
        for w1, b1, m1, w2, b2, m2 in self.res_blocks:
            pad1 = np.zeros((11, 11, c), dtype=np.int32)
            pad1[1:10, 1:10, :] = act.astype(np.int32)
            buf1 = np.zeros((9, 9, c), dtype=np.int8)
            for r in range(9):
                for col in range(9):
                    patch = pad1[r : r + 3, col : col + 3, :]
                    acc = b1 + np.sum(patch * w1, axis=(1, 2, 3))
                    scaled = np.round(acc * m1).astype(np.int32)
                    buf1[r, col] = np.clip(scaled, 0, 127).astype(np.int8)

            pad2 = np.zeros((11, 11, c), dtype=np.int32)
            pad2[1:10, 1:10, :] = buf1.astype(np.int32)
            buf2 = np.zeros((9, 9, c), dtype=np.int8)
            for r in range(9):
                for col in range(9):
                    patch = pad2[r : r + 3, col : col + 3, :]
                    acc = b2 + np.sum(patch * w2, axis=(1, 2, 3))
                    scaled = np.round(acc * m2).astype(np.int32)
                    skip = act[r, col].astype(np.int32)
                    buf2[r, col] = np.clip(scaled + skip, 0, 127).astype(np.int8)

            act = buf2

        # 3. Policy Head
        flat_p = np.zeros((2, 9, 9), dtype=np.int8)
        for oc in range(2):
            acc = self.conv_p_b[oc] + np.sum(act.astype(np.int32) * self.conv_p_w[oc], axis=-1)
            scaled = np.round(acc * self.conv_p_m).astype(np.int32)
            flat_p[oc] = np.clip(scaled, 0, 127).astype(np.int8)

        flat_p_vec = flat_p.reshape(2 * 81).astype(np.int32)
        policy_acc = self.fc_p_b + np.dot(self.fc_p_w, flat_p_vec)
        policy_logits = policy_acc * self.fc_p_m
        policy_logits[~legal_mask_bool] = -1e9

        # 4. Value Head
        flat_v = np.zeros((9, 9), dtype=np.int8)
        acc_v = self.conv_v_b[0] + np.sum(act.astype(np.int32) * self.conv_v_w[0], axis=-1)
        scaled_v = np.round(acc_v * self.conv_v_m).astype(np.int32)
        flat_v = np.clip(scaled_v, 0, 127).astype(np.int8).reshape(81).astype(np.int32)

        hidden_v = self.fc_v1_b + np.dot(self.fc_v1_w, flat_v)
        hidden_v_q = np.clip(np.round(hidden_v * self.fc_v1_m), 0, 127).astype(np.int8).astype(np.int32)

        raw_val = float(self.fc_v2_b[0] + np.dot(self.fc_v2_w[0], hidden_v_q)) * self.fc_v2_m
        val = np.tanh(raw_val)

        return policy_logits, float(val)


def run_int8_parity_test(num_positions: int = 100):
    print("=" * 60)
    print("  AlphaZero INT8 Quantization Parity Verification")
    print("=" * 60)

    # 1. Load PyTorch model
    champion_path = "ml/champion.pt" if os.path.exists("ml/champion.pt") else "ml/model_checkpoint.pt"
    if not os.path.exists(champion_path):
        print(f"Skipping: No checkpoint found at {champion_path}")
        return

    pt_model = AlphaZeroResNet().to("cpu")
    ckpt = torch.load(champion_path, map_location="cpu")
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()
    print(f"Loaded PyTorch reference model from {champion_path}")

    # 2. Load INT8 binary model
    int8_bin = "uttt/src/nn/model_int8.bin"
    if not os.path.exists(int8_bin):
        print(f"Error: {int8_bin} not found")
        sys.exit(1)

    int8_model = Int8ModelEmulator(int8_bin)
    print(f"Loaded INT8 binary model from {int8_bin}")

    # 3. Load test positions from replay buffer
    buffer = ContiguousStratifiedReplayBuffer(capacity=100000)
    buffer_path = "ml/replay_buffer_checkpoint.npz"
    if os.path.exists(buffer_path):
        buffer.load(buffer_path)
        obs, masks, _, _ = buffer.sample_batch(batch_size=num_positions)
    else:
        print("Synthesizing diverse random positions for testing...")
        obs = np.random.randint(0, 2, size=(num_positions, 5, 9, 9), dtype=np.uint8).astype(np.float32)
        masks = np.random.randint(0, 2, size=(num_positions, 81), dtype=np.uint8).astype(np.float32)
        masks[:, 40] = 1.0  # guarantee at least one legal move

    print(f"Evaluating {num_positions} positions across PyTorch FP32 and INT8 binary...")

    cos_sims = []
    val_diffs = []
    top1_matches = 0

    with torch.no_grad():
        for i in range(num_positions):
            t_obs = torch.from_numpy(obs[i : i + 1]).float()
            t_mask = torch.from_numpy(masks[i : i + 1]).float()

            pt_logits, pt_val = pt_model(t_obs, t_mask)
            pt_p = F.softmax(pt_logits[0], dim=-1).numpy()
            pt_v = float(pt_val[0, 0].numpy())

            # Convert (5, 9, 9) CHW to (9, 9, 5) HWC
            hwc_obs = np.transpose(obs[i], (1, 2, 0)).astype(np.int8)
            mask_bool = masks[i] > 0.5

            int8_logits, int8_v = int8_model.forward(hwc_obs, mask_bool)

            # Masked softmax for INT8
            int8_logits_clamped = int8_logits.copy()
            int8_logits_clamped[~mask_bool] = -1e9
            exp_l = np.exp(int8_logits_clamped - np.max(int8_logits_clamped))
            int8_p = exp_l / np.sum(exp_l)

            # Cosine similarity
            dot = np.dot(pt_p, int8_p)
            norm = np.linalg.norm(pt_p) * np.linalg.norm(int8_p)
            cos = dot / max(norm, 1e-8)
            cos_sims.append(cos)

            # Value difference
            val_diffs.append(abs(pt_v - int8_v))

            # Top-1 move agreement
            if np.argmax(pt_p) == np.argmax(int8_p):
                top1_matches += 1

    avg_cos = np.mean(cos_sims)
    min_cos = np.min(cos_sims)
    avg_v_diff = np.mean(val_diffs)
    max_v_diff = np.max(val_diffs)
    top1_acc = top1_matches / num_positions

    print(f"\nResults over {num_positions} positions:")
    print(f"  Policy Cosine Similarity: Mean = {avg_cos:.4f}, Min = {min_cos:.4f}")
    print(f"  Top-1 Move Agreement:    {top1_acc*100:.1f}%")
    print(f"  Value Absolute Error:    Mean = {avg_v_diff:.4f}, Max = {max_v_diff:.4f}")

    assert avg_cos >= 0.90, f"Policy cosine similarity too low: {avg_cos:.4f} < 0.90"
    assert avg_v_diff <= 0.15, f"Value error too high: {avg_v_diff:.4f} > 0.15"

    print("\n✅ INT8 QUANTIZATION PARITY VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    n = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    run_int8_parity_test(n)
