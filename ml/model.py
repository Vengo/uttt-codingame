"""
AlphaZero Dual-Headed ResNet for Ultimate Tic-Tac-Toe
Architecture:
- Spatial input: 5 x 9 x 9 (NCHW)
- Shared trunk: Conv2d(5, 64, k=3, p=1) + BatchNorm2d + ReLU
                followed by 4 Residual Blocks (Conv-BN-ReLU-Conv-BN + skip)
- Policy head:  Conv2d(64, 2, k=1) + BatchNorm2d + ReLU + Flatten + Linear(162, 81)
                + legal move masking (-1e9 for illegal moves)
- Value head:   Conv2d(64, 1, k=1) + BatchNorm2d + ReLU + Flatten + Linear(81, 64)
                + ReLU + Linear(64, 1) + Tanh -> [-1.0, 1.0]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class AlphaZeroResNet(nn.Module):
    def __init__(self, num_channels: int = 64, num_res_blocks: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # Shared trunk
        self.conv0 = nn.Conv2d(5, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(num_channels)
        self.res_blocks = nn.ModuleList([ResBlock(num_channels) for _ in range(num_res_blocks)])

        # Policy head: Conv(64, 2, 1) + BN + ReLU + Flatten + Linear(162, 81)
        self.conv_p = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.bn_p = nn.BatchNorm2d(2)
        self.fc_p = nn.Linear(2 * 9 * 9, 81)

        # Value head: Conv(64, 1, 1) + BN + ReLU + Flatten + Linear(81, 64) + ReLU + Linear(64, 1) + Tanh
        self.conv_v = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.bn_v = nn.BatchNorm2d(1)
        self.fc_v1 = nn.Linear(1 * 9 * 9, 64)
        self.fc_v2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor = None):
        """
        x: tensor of shape (B, 5, 9, 9)
        legal_mask: optional tensor of shape (B, 81) with 1 for legal, 0 for illegal
        Returns:
            logits: tensor of shape (B, 81) (masked with -1e9 if legal_mask is provided)
            value:  tensor of shape (B, 1) in [-1.0, 1.0]
        """
        # Shared trunk
        h = F.relu(self.bn0(self.conv0(x)))
        for block in self.res_blocks:
            h = block(h)

        # Policy head
        p = F.relu(self.bn_p(self.conv_p(h)))
        p = self.fc_p(p.view(p.size(0), -1))
        if legal_mask is not None:
            fill_val = -1e4 if p.dtype == torch.float16 else -1e9
            p = torch.where(legal_mask.bool(), p, torch.full_like(p, fill_val))

        # Value head
        v = F.relu(self.bn_v(self.conv_v(h)))
        v = F.relu(self.fc_v1(v.view(v.size(0), -1)))
        v = torch.tanh(self.fc_v2(v))

        return p, v


# Spatial coordinate conversion helpers
def move_index_to_row_col(move_index: int) -> tuple[int, int]:
    """Convert move_index (0..80) to 2D board coordinates (row, col) in [0..8]."""
    subboard = move_index // 9
    subsquare = move_index % 9
    row = (subboard // 3) * 3 + (subsquare // 3)
    col = (subboard % 3) * 3 + (subsquare % 3)
    return row, col


def row_col_to_move_index(row: int, col: int) -> int:
    """Convert 2D board coordinates (row, col) in [0..8] to move_index (0..80)."""
    subboard = 3 * (row // 3) + (col // 3)
    subsquare = 3 * (row % 3) + (col % 3)
    return 9 * subboard + subsquare


def encode_board(
    xbb: list[int],
    obb: list[int],
    x_supers: int,
    o_supers: int,
    last_super_index: int,
    x_to_move: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode UltimateTicTacToeBoard state into a (5, 9, 9) float32 tensor
    and an (81,) float32 legal move mask.

    Channels:
      0: Active player marks
      1: Opponent marks
      2: Active player won subboards (across all 9 cells)
      3: Opponent won subboards (across all 9 cells)
      4: Legal target subboard mask (across all 9 cells of legal subboards)
    """
    tensor = np.zeros((5, 9, 9), dtype=np.float32)
    legal_mask = np.zeros(81, dtype=np.float32)

    active_bb = xbb if x_to_move else obb
    opp_bb = obb if x_to_move else xbb
    active_supers = x_supers if x_to_move else o_supers
    opp_supers = o_supers if x_to_move else x_supers

    # Calculate tied subboards mask: subboards where all 9 cells are full and neither won
    won_mask = x_supers | o_supers
    tied_mask = 0
    for sb in range(9):
        if (won_mask & (1 << sb)) == 0 and ((xbb[sb] | obb[sb]) == 0x1FF):
            tied_mask |= (1 << sb)
    complete_bbs = won_mask | tied_mask

    # Determine legal target subboards
    if last_super_index >= 9 or (complete_bbs & (1 << last_super_index)) != 0:
        target_subboard_mask = (~complete_bbs) & 0x1FF
    else:
        target_subboard_mask = 1 << last_super_index

    # Populate 5x9x9 tensor
    for sb in range(9):
        sb_r = sb // 3
        sb_c = sb % 3
        is_legal_target = (target_subboard_mask & (1 << sb)) != 0
        is_active_won = (active_supers & (1 << sb)) != 0
        is_opp_won = (opp_supers & (1 << sb)) != 0

        for sq in range(9):
            sq_r = sq // 3
            sq_c = sq % 3
            r = sb_r * 3 + sq_r
            c = sb_c * 3 + sq_c
            move_idx = 9 * sb + sq

            # Channel 0: Active player mark
            if (active_bb[sb] & (1 << sq)) != 0:
                tensor[0, r, c] = 1.0

            # Channel 1: Opponent mark
            if (opp_bb[sb] & (1 << sq)) != 0:
                tensor[1, r, c] = 1.0

            # Channel 2: Active player won subboard
            if is_active_won:
                tensor[2, r, c] = 1.0

            # Channel 3: Opponent won subboard
            if is_opp_won:
                tensor[3, r, c] = 1.0

            # Channel 4: Legal target subboard mask
            if is_legal_target:
                tensor[4, r, c] = 1.0

            # Legal move mask: cell is empty and in a legal target subboard
            if is_legal_target and ((xbb[sb] | obb[sb]) & (1 << sq)) == 0:
                legal_mask[move_idx] = 1.0

    return tensor, legal_mask
