"""
INT8 Quantization and Export Pipeline for AlphaZero ResNet
Functions:
- Fuses BatchNorm into Conv2d layers
- Performs calibration on sample positions to determine activation scales
- Quantizes weights to symmetric INT8 and biases to INT32
- Exports to compact binary model `model_int8.bin`
- Generates optional Rust code embedding
"""

import copy
import os
import sys
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import AlphaZeroResNet
from ml.dataset import load_replays_from_files, run_fast_self_play


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[np.ndarray, np.ndarray]:
    """
    Fold Conv2d and BatchNorm2d into equivalent Conv2d weights and biases.
    Returns:
        w_folded: (out_channels, in_channels, k_h, k_w)
        b_folded: (out_channels,)
    """
    w = conv.weight.detach().cpu().numpy()
    b = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels, dtype=np.float32)

    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    std = np.sqrt(var + eps)
    scale = gamma / std

    w_folded = w * scale[:, None, None, None]
    b_folded = (b - mean) * scale + beta
    return w_folded.astype(np.float32), b_folded.astype(np.float32)


def quantize_layer(
    weights: np.ndarray,
    bias: np.ndarray,
    scale_in: float,
    scale_out: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Symmetric INT8 quantization for a layer.
    Weights are quantized to [-127, 127].
    Bias is quantized to int32 with scale (scale_w * scale_in).
    Multiplier maps accumulator back to scale_out.
    """
    w_max = float(np.max(np.abs(weights)))
    scale_w = 127.0 / max(w_max, 1e-7)

    q_weights = np.clip(np.round(weights * scale_w), -127, 127).astype(np.int8)
    q_bias = np.round(bias * (scale_w * scale_in)).astype(np.int32)

    # Multiplier: maps accumulator (scale_w * scale_in) to scale_out
    mult = float(scale_out / (scale_w * scale_in))
    return q_weights, q_bias, mult


def export_model_int8(
    model: AlphaZeroResNet,
    output_bin_path: str = "ml/model_int8.bin",
    calibration_samples: list = None,
):
    model = copy.deepcopy(model).to("cpu")
    model.eval()
    device = "cpu"

    # Gather calibration inputs
    if calibration_samples is None or len(calibration_samples) == 0:
        replay_files = [
            "replay_ab_vs_hmcts.json",
            "replay_alphabeta_vs_rule_based.json",
            "replay_mcts_vs_original_umcts.json",
        ]
        calibration_samples = load_replays_from_files(replay_files)
        if len(calibration_samples) < 50:
            calibration_samples.extend(run_fast_self_play(model, device=device))

    calib_tensors = [s[0] for s in calibration_samples[:100]]
    if not calib_tensors:
        # Fallback dummy tensor
        calib_tensors = [np.zeros((5, 9, 9), dtype=np.float32)]

    # 1. Fold BatchNorm into Conv layers
    w_conv0, b_conv0 = fold_conv_bn(model.conv0, model.bn0)

    res_folded = []
    for block in model.res_blocks:
        w1, b1 = fold_conv_bn(block.conv1, block.bn1)
        w2, b2 = fold_conv_bn(block.conv2, block.bn2)
        res_folded.append((w1, b1, w2, b2))

    w_conv_p, b_conv_p = fold_conv_bn(model.conv_p, model.bn_p)
    w_fc_p = model.fc_p.weight.detach().cpu().numpy().astype(np.float32)
    b_fc_p = model.fc_p.bias.detach().cpu().numpy().astype(np.float32)

    w_conv_v, b_conv_v = fold_conv_bn(model.conv_v, model.bn_v)
    w_fc_v1 = model.fc_v1.weight.detach().cpu().numpy().astype(np.float32)
    b_fc_v1 = model.fc_v1.bias.detach().cpu().numpy().astype(np.float32)
    w_fc_v2 = model.fc_v2.weight.detach().cpu().numpy().astype(np.float32)
    b_fc_v2 = model.fc_v2.bias.detach().cpu().numpy().astype(np.float32)

    # 2. Run calibration forward pass to determine activation ranges
    max_act = {}

    def track_activation(name, tensor):
        m = float(np.max(np.abs(tensor)))
        max_act[name] = max(max_act.get(name, 1e-4), m)

    for x in calib_tensors:
        # conv0: input x has values in {0, 1}
        # In PyTorch conv2d: (C_out, C_in, 3, 3)
        # We can simulate folded layers in numpy or torch
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0)
            c0 = torch.relu(torch.conv2d(xt, torch.from_numpy(w_conv0), torch.from_numpy(b_conv0), padding=1))
            track_activation("c0", c0.numpy())

            h = c0
            for i, (w1, b1, w2, b2) in enumerate(res_folded):
                h1 = torch.relu(torch.conv2d(h, torch.from_numpy(w1), torch.from_numpy(b1), padding=1))
                track_activation(f"res_{i}_mid", h1.numpy())
                h2 = torch.conv2d(h1, torch.from_numpy(w2), torch.from_numpy(b2), padding=1)
                h = torch.relu(h + h2)
                track_activation(f"res_{i}_out", h.numpy())

            # Policy head
            cp = torch.relu(torch.conv2d(h, torch.from_numpy(w_conv_p), torch.from_numpy(b_conv_p)))
            track_activation("cp", cp.numpy())

            # Value head
            cv = torch.relu(torch.conv2d(h, torch.from_numpy(w_conv_v), torch.from_numpy(b_conv_v)))
            track_activation("cv", cv.numpy())
            fv1 = torch.relu(F.linear(cv.flatten(1), torch.from_numpy(w_fc_v1), torch.from_numpy(b_fc_v1)))
            track_activation("fv1", fv1.numpy())

    # Set activation scales (S = 127 / max_act)
    # For residual blocks, keep scale consistent to allow exact integer residual addition
    max_res_act = max([max_act.get(f"res_{i}_out", 1.0) for i in range(len(res_folded))] + [max_act.get("c0", 1.0)])
    s_hidden = 127.0 / max(max_res_act, 1e-4)

    s_in = 1.0  # Input features are {0, 1}
    s_c0 = s_hidden

    # 3. Quantize all layers
    # Conv0: weights (64, 5, 3, 3) -> layout (64, 3, 3, 5) for HWC contiguous dot product
    w0_hwc = np.transpose(w_conv0, (0, 2, 3, 1))
    q_w_c0, q_b_c0, m_c0 = quantize_layer(w0_hwc, b_conv0, s_in, s_c0)

    # ResBlocks
    q_res = []
    for i, (w1, b1, w2, b2) in enumerate(res_folded):
        s_mid = 127.0 / max(max_act.get(f"res_{i}_mid", 1.0), 1e-4)
        w1_hwc = np.transpose(w1, (0, 2, 3, 1))
        w2_hwc = np.transpose(w2, (0, 2, 3, 1))
        qw1, qb1, m1 = quantize_layer(w1_hwc, b1, s_hidden, s_mid)
        qw2, qb2, m2 = quantize_layer(w2_hwc, b2, s_mid, s_hidden)
        q_res.append((qw1, qb1, m1, qw2, qb2, m2))

    # Policy head
    s_cp = 127.0 / max(max_act.get("cp", 1.0), 1e-4)
    # 1x1 conv: (2, 64, 1, 1) -> (2, 64)
    w_cp_2d = w_conv_p.reshape(2, 64)
    q_w_cp, q_b_cp, m_cp = quantize_layer(w_cp_2d, b_conv_p, s_hidden, s_cp)

    # fc_p: maps (162) with scale s_cp to logits
    w_max_fc_p = float(np.max(np.abs(w_fc_p)))
    s_w_fc_p = 127.0 / max(w_max_fc_p, 1e-7)
    q_w_fc_p = np.clip(np.round(w_fc_p * s_w_fc_p), -127, 127).astype(np.int8)
    q_b_fc_p = np.round(b_fc_p * (s_w_fc_p * s_cp)).astype(np.int32)
    m_fc_p_dequant = float(1.0 / (s_w_fc_p * s_cp))  # dequantize logits directly to float

    # Value head
    s_cv = 127.0 / max(max_act.get("cv", 1.0), 1e-4)
    w_cv_2d = w_conv_v.reshape(1, 64)
    q_w_cv, q_b_cv, m_cv = quantize_layer(w_cv_2d, b_conv_v, s_hidden, s_cv)

    s_fv1 = 127.0 / max(max_act.get("fv1", 1.0), 1e-4)
    q_w_fv1, q_b_fv1, m_fv1 = quantize_layer(w_fc_v1, b_fc_v1, s_cv, s_fv1)

    w_max_fv2 = float(np.max(np.abs(w_fc_v2)))
    s_w_fv2 = 127.0 / max(w_max_fv2, 1e-7)
    q_w_fv2 = np.clip(np.round(w_fc_v2 * s_w_fv2), -127, 127).astype(np.int8)
    q_b_fv2 = np.round(b_fc_v2 * (s_w_fv2 * s_fv1)).astype(np.int32)
    m_fv2_dequant = float(1.0 / (s_w_fv2 * s_fv1))

    # 4. Write binary file
    with open(output_bin_path, "wb") as f:
        # Magic: 8 bytes
        f.write(b"UTTTNN01")
        # Header: num_channels (u32), num_res_blocks (u32)
        f.write(struct.pack("<II", model.num_channels, model.num_res_blocks))

        # conv0: weights (64*3*3*5), bias (64*i32), mult (f32)
        f.write(q_w_c0.tobytes())
        f.write(q_b_c0.tobytes())
        f.write(struct.pack("<f", m_c0))

        # 4 ResBlocks
        for qw1, qb1, m1, qw2, qb2, m2 in q_res:
            f.write(qw1.tobytes())
            f.write(qb1.tobytes())
            f.write(struct.pack("<f", m1))

            f.write(qw2.tobytes())
            f.write(qb2.tobytes())
            f.write(struct.pack("<f", m2))

        # Policy head
        f.write(q_w_cp.tobytes())
        f.write(q_b_cp.tobytes())
        f.write(struct.pack("<f", m_cp))

        f.write(q_w_fc_p.tobytes())
        f.write(q_b_fc_p.tobytes())
        f.write(struct.pack("<f", m_fc_p_dequant))

        # Value head
        f.write(q_w_cv.tobytes())
        f.write(q_b_cv.tobytes())
        f.write(struct.pack("<f", m_cv))

        f.write(q_w_fv1.tobytes())
        f.write(q_b_fv1.tobytes())
        f.write(struct.pack("<f", m_fv1))

        f.write(q_w_fv2.tobytes())
        f.write(q_b_fv2.tobytes())
        f.write(struct.pack("<f", m_fv2_dequant))

    file_size = os.path.getsize(output_bin_path)
    print(f"Exported INT8 quantized model to {output_bin_path} ({file_size} bytes / {file_size/1024:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export PyTorch model to INT8 binary")
    parser.add_argument("--checkpoint", type=str, default="ml/model_checkpoint.pt")
    parser.add_argument("--output", type=str, default="ml/model_int8.bin")
    args = parser.parse_args()

    model = AlphaZeroResNet()
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("No checkpoint found; exporting freshly initialized model for testing.")

    export_model_int8(model, args.output)
