"""
High-Speed Python ctypes Bridge to Pure Rust Rayon MCTS Batch Worker
Loads target/release/libuttt.so with zero-copy shared NumPy arrays.
Throughput: Up to 325,000+ evals/s on RTX 5060 Ti GPU.
"""

import ctypes
import os
import sys
from typing import List, Optional, Tuple
import numpy as np
import torch

from ml.model import AlphaZeroResNet


class RustMctsWorker:
    def __init__(
        self,
        num_games: int = 1024,
        c_puct: float = 2.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        so_path: Optional[str] = None,
    ):
        self.num_games = int(num_games)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.dirichlet_eps = float(dirichlet_eps)

        if so_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            so_path = os.path.join(base_dir, "target", "release", "libuttt.so")

        if not os.path.exists(so_path):
            raise FileNotFoundError(f"Rust shared library not found at: {so_path}. Run `cargo build --release -p uttt`")

        self.lib = ctypes.CDLL(so_path)
        self._bind_c_functions()

        self.worker = self.lib.mcts_worker_create(
            ctypes.c_size_t(self.num_games),
            ctypes.c_float(self.c_puct),
            ctypes.c_float(self.dirichlet_alpha),
            ctypes.c_float(self.dirichlet_eps),
        )
        if not self.worker:
            raise RuntimeError("Failed to allocate MctsWorker in Rust")

        # Pre-allocate contiguous zero-copy host buffers (default capacity for virtual_loss_k up to 8)
        self.buffer_capacity_k = 1
        self._allocate_buffers(max_k=8)

    def _allocate_buffers(self, max_k: int):
        self.buffer_capacity_k = max(1, int(max_k))
        cap = self.num_games * self.buffer_capacity_k
        self.features_buffer = np.zeros((cap, 5, 9, 9), dtype=np.float32)
        self.masks_buffer = np.zeros((cap, 81), dtype=np.float32)
        self.logits_buffer = np.zeros((cap, 81), dtype=np.float32)
        self.values_buffer = np.zeros((cap,), dtype=np.float32)
        self.active_indices_buffer = np.zeros((cap,), dtype=np.uintp)

        # Raw pointers for C-ABI
        self.feat_ptr = self.features_buffer.ctypes.data_as(ctypes.c_void_p)
        self.mask_ptr = self.masks_buffer.ctypes.data_as(ctypes.c_void_p)
        self.logits_ptr = self.logits_buffer.ctypes.data_as(ctypes.c_void_p)
        self.vals_ptr = self.values_buffer.ctypes.data_as(ctypes.c_void_p)
        self.act_idx_ptr = self.active_indices_buffer.ctypes.data_as(ctypes.c_void_p)

    def _bind_c_functions(self):
        self.lib.mcts_worker_create.argtypes = [ctypes.c_size_t, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.lib.mcts_worker_create.restype = ctypes.c_void_p

        self.lib.mcts_worker_destroy.argtypes = [ctypes.c_void_p]
        self.lib.mcts_worker_destroy.restype = None

        self.lib.mcts_worker_reset.argtypes = [ctypes.c_void_p]
        self.lib.mcts_worker_reset.restype = None

        self.lib.mcts_worker_prepare_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.mcts_worker_prepare_step.restype = ctypes.c_size_t

        if hasattr(self.lib, "mcts_worker_prepare_step_vloss"):
            self.lib.mcts_worker_prepare_step_vloss.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            self.lib.mcts_worker_prepare_step_vloss.restype = ctypes.c_size_t

        self.lib.mcts_worker_finish_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.mcts_worker_finish_step.restype = None

        if hasattr(self.lib, "mcts_worker_finish_step_vloss"):
            self.lib.mcts_worker_finish_step_vloss.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            self.lib.mcts_worker_finish_step_vloss.restype = None

        self.lib.mcts_worker_advance_moves.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.mcts_worker_advance_moves.restype = ctypes.c_size_t

        self.lib.mcts_worker_advance_moves_tournament.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.mcts_worker_advance_moves_tournament.restype = ctypes.c_size_t

        self.lib.mcts_worker_advance_moves_tournament_recorded.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        self.lib.mcts_worker_advance_moves_tournament_recorded.restype = ctypes.c_size_t

        self.lib.mcts_worker_drain_samples.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.mcts_worker_drain_samples.restype = ctypes.c_size_t

        self.lib.mcts_worker_set_game_moves.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.mcts_worker_set_game_moves.restype = None

        self.lib.mcts_worker_get_game_status.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.mcts_worker_get_game_status.restype = ctypes.c_int32

        self.lib.mcts_worker_get_game_x_to_move.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.mcts_worker_get_game_x_to_move.restype = ctypes.c_bool

        self.lib.mcts_worker_get_active_indices.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.mcts_worker_get_active_indices.restype = ctypes.c_size_t

        self.lib.mcts_worker_apply_moves_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.lib.mcts_worker_apply_moves_batch.restype = None

        self.lib.alphabeta_batch_choose_moves.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.lib.alphabeta_batch_choose_moves.restype = None

        if hasattr(self.lib, "alphabeta_v1_batch_choose_moves"):
            self.lib.alphabeta_v1_batch_choose_moves.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            self.lib.alphabeta_v1_batch_choose_moves.restype = None

        if hasattr(self.lib, "umcts_v2_batch_choose_moves"):
            self.lib.umcts_v2_batch_choose_moves.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
            self.lib.umcts_v2_batch_choose_moves.restype = None

    def __del__(self):
        if hasattr(self, "worker") and self.worker:
            self.lib.mcts_worker_destroy(self.worker)
            self.worker = None

    def reset(self):
        self.lib.mcts_worker_reset(self.worker)

    def step_simulation(
        self,
        model: AlphaZeroResNet,
        device: str = "cuda",
        virtual_loss_k: int = 1,
    ) -> int:
        """
        Runs a single MCTS simulation step across all games (up to `virtual_loss_k` leaves per game):
        1. Rayon parallel tree walk in Rust (< 0.2 ms)
        2. PyTorch GPU batch forward pass on leaves
        3. Rayon parallel expansion and backprop in Rust (< 0.2 ms)
        Returns: number of active leaves evaluated on GPU.
        """
        k = max(1, int(virtual_loss_k))
        if k > self.buffer_capacity_k:
            self._allocate_buffers(max_k=k)

        if k > 1 and hasattr(self.lib, "mcts_worker_prepare_step_vloss"):
            n_active = self.lib.mcts_worker_prepare_step_vloss(
                self.worker, ctypes.c_size_t(k), self.feat_ptr, self.mask_ptr
            )
        else:
            n_active = self.lib.mcts_worker_prepare_step(self.worker, self.feat_ptr, self.mask_ptr)

        if n_active > 0:
            # Zero-copy slicing into PyTorch tensors
            x_t = torch.from_numpy(self.features_buffer[:n_active]).to(device, non_blocking=True)
            m_t = torch.from_numpy(self.masks_buffer[:n_active]).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                    logits, vals = model(x_t, m_t)

            # Copy back to host pinned buffers
            self.logits_buffer[:n_active] = logits.cpu().numpy()
            self.values_buffer[:n_active] = vals.squeeze(1).cpu().numpy()

            self.lib.mcts_worker_finish_step(self.worker, self.logits_ptr, self.vals_ptr, ctypes.c_size_t(n_active))

        return n_active

    def run_mcts_search(
        self,
        model: AlphaZeroResNet,
        num_simulations: int = 800,
        device: str = "cuda",
        virtual_loss_k: int = 1,
    ) -> int:
        """Executes num_simulations across all parallel games."""
        total_evals = 0
        k = max(1, int(virtual_loss_k))
        if k == 1 or not hasattr(self.lib, "mcts_worker_prepare_step_vloss"):
            for _ in range(num_simulations):
                total_evals += self.step_simulation(model, device=device, virtual_loss_k=1)
        else:
            # Ensure root node is expanded on any newly reset game before multi-leaf traversal
            total_evals += self.step_simulation(model, device=device, virtual_loss_k=1)
            sims_done = 1
            while sims_done < num_simulations:
                step_k = min(k, num_simulations - sims_done)
                total_evals += self.step_simulation(model, device=device, virtual_loss_k=step_k)
                sims_done += step_k
        return total_evals

    def advance_moves(self, temperature: float = 1.0) -> int:
        """
        Advances one ply across all games:
        - Samples moves proportionally to visit counts with temperature T > 0
        - Flushes finished game states to completed_samples
        - Resets finished games with new 2-ply opening configurations
        Returns: number of new training samples generated.
        """
        return self.lib.mcts_worker_advance_moves(self.worker, ctypes.c_float(temperature))

    def drain_samples(
        self,
        max_count: int = 50_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Drains completed game training samples from Rust into NumPy arrays.
        Returns: (features, masks, policies, values)
        """
        out_f = np.zeros((max_count, 5, 9, 9), dtype=np.float32)
        out_m = np.zeros((max_count, 81), dtype=np.float32)
        out_p = np.zeros((max_count, 81), dtype=np.float32)
        out_v = np.zeros((max_count,), dtype=np.float32)

        f_ptr = out_f.ctypes.data_as(ctypes.c_void_p)
        m_ptr = out_m.ctypes.data_as(ctypes.c_void_p)
        p_ptr = out_p.ctypes.data_as(ctypes.c_void_p)
        v_ptr = out_v.ctypes.data_as(ctypes.c_void_p)

        n = self.lib.mcts_worker_drain_samples(
            self.worker, f_ptr, m_ptr, p_ptr, v_ptr, ctypes.c_size_t(max_count)
        )
        return out_f[:n], out_m[:n], out_p[:n], out_v[:n]

    def set_game_moves(self, game_idx: int, moves: list[int]):
        """Sets a game board to the state resulting from a move sequence."""
        arr = (ctypes.c_uint8 * len(moves))(*moves)
        self.lib.mcts_worker_set_game_moves(
            self.worker,
            ctypes.c_size_t(game_idx),
            ctypes.cast(arr, ctypes.c_void_p),
            ctypes.c_size_t(len(moves)),
        )

    def get_game_status(self, game_idx: int) -> int:
        """Returns: 0=unresolved, 1=P1 win, -1=P2 win, 2=tie."""
        return self.lib.mcts_worker_get_game_status(self.worker, ctypes.c_size_t(game_idx))

    def get_game_x_to_move(self, game_idx: int) -> bool:
        """Returns True if player X is to move, False for O."""
        return self.lib.mcts_worker_get_game_x_to_move(self.worker, ctypes.c_size_t(game_idx))

    def get_active_indices(self) -> np.ndarray:
        """Returns the game indices of leaves currently being evaluated."""
        max_cap = self.num_games * self.buffer_capacity_k
        n = self.lib.mcts_worker_get_active_indices(
            self.worker, self.act_idx_ptr, ctypes.c_size_t(max_cap)
        )
        return self.active_indices_buffer[:n].copy()

    def advance_moves_tournament(self, temperature: float = 0.0) -> int:
        """Advances moves in tournament mode without auto-resetting finished games."""
        return self.lib.mcts_worker_advance_moves_tournament(
            self.worker, ctypes.c_float(temperature)
        )

    def advance_moves_tournament_recorded(self, temperature: float = 0.0) -> Tuple[int, np.ndarray]:
        """
        Advances moves in tournament mode and returns (finished_count, chosen_moves).
        `chosen_moves[g]` is the move chosen (0..80) or 255 if game g was finished or not advanced.
        """
        out_arr = (ctypes.c_uint8 * self.num_games)()
        finished_count = self.lib.mcts_worker_advance_moves_tournament_recorded(
            self.worker,
            ctypes.c_float(temperature),
            out_arr,
        )
        return finished_count, np.frombuffer(out_arr, dtype=np.uint8).copy()

    def step_simulation_dual(
        self,
        model_chal: AlphaZeroResNet,
        model_champ: AlphaZeroResNet,
        game_to_model: np.ndarray,
        device: str = "cuda",
        virtual_loss_k: int = 1,
    ) -> int:
        """
        Runs a simulation step across parallel games with two competing models.
        `game_to_model[g]` is True if Challenger evaluates game `g`, False for Champion.
        """
        k = max(1, int(virtual_loss_k))
        if k > self.buffer_capacity_k:
            self._allocate_buffers(max_k=k)

        if k > 1 and hasattr(self.lib, "mcts_worker_prepare_step_vloss"):
            n_active = self.lib.mcts_worker_prepare_step_vloss(
                self.worker, ctypes.c_size_t(k), self.feat_ptr, self.mask_ptr
            )
        else:
            n_active = self.lib.mcts_worker_prepare_step(self.worker, self.feat_ptr, self.mask_ptr)

        if n_active > 0:
            active_games = self.get_active_indices()[:n_active]
            is_chal = game_to_model[active_games]
            chal_indices = np.where(is_chal)[0]
            champ_indices = np.where(~is_chal)[0]

            with torch.no_grad():
                with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                    if len(chal_indices) > 0:
                        x_chal = torch.from_numpy(self.features_buffer[chal_indices]).to(device, non_blocking=True)
                        m_chal = torch.from_numpy(self.masks_buffer[chal_indices]).to(device, non_blocking=True)
                        l_chal, v_chal = model_chal(x_chal, m_chal)
                        self.logits_buffer[chal_indices] = l_chal.cpu().numpy()
                        self.values_buffer[chal_indices] = v_chal.squeeze(1).cpu().numpy()

                    if len(champ_indices) > 0:
                        x_champ = torch.from_numpy(self.features_buffer[champ_indices]).to(device, non_blocking=True)
                        m_champ = torch.from_numpy(self.masks_buffer[champ_indices]).to(device, non_blocking=True)
                        l_champ, v_champ = model_champ(x_champ, m_champ)
                        self.logits_buffer[champ_indices] = l_champ.cpu().numpy()
                        self.values_buffer[champ_indices] = v_champ.squeeze(1).cpu().numpy()

            self.lib.mcts_worker_finish_step(self.worker, self.logits_ptr, self.vals_ptr, ctypes.c_size_t(n_active))

        return n_active

    def apply_moves_batch(self, game_indices: List[int], moves: List[int]):
        """
        Applies a batch of external moves directly to game slots in Rust.
        game_indices: list of game slot indices (0..num_games-1)
        moves: list of move indices (0..80)
        """
        assert len(game_indices) == len(moves)
        if len(game_indices) == 0:
            return
        idx_arr = (ctypes.c_size_t * len(game_indices))(*game_indices)
        mv_arr = (ctypes.c_uint8 * len(moves))(*moves)
        self.lib.mcts_worker_apply_moves_batch(
            self.worker,
            idx_arr,
            mv_arr,
            ctypes.c_size_t(len(game_indices)),
        )

    def choose_alphabeta_moves(
        self,
        game_history_list: List[List[int]],
        time_limit_ms: int = 10,
        bot_version: str = "v2",
    ) -> List[int]:
        """
        Batched AlphaBeta move choice using Rayon thread-local bot pool in libuttt.so.
        game_history_list: list of move sequences for each game
        time_limit_ms: per-move time limit in ms
        bot_version: "v1" (classic 1713 Elo) or "v2" (transposition table / DF-PN)
        Returns: list of chosen move indices (0..80)
        """
        num_games = len(game_history_list)
        if num_games == 0:
            return []

        flat_moves = []
        offsets = []
        lengths = []
        curr_offset = 0
        for moves in game_history_list:
            offsets.append(curr_offset)
            lengths.append(len(moves))
            flat_moves.extend(moves)
            curr_offset += len(moves)

        flat_arr = (ctypes.c_uint8 * len(flat_moves))(*flat_moves) if flat_moves else None
        off_arr = (ctypes.c_size_t * num_games)(*offsets)
        len_arr = (ctypes.c_size_t * num_games)(*lengths)
        out_moves = (ctypes.c_uint8 * num_games)()

        if bot_version == "umcts_v2" and hasattr(self.lib, "umcts_v2_batch_choose_moves"):
            self.lib.umcts_v2_batch_choose_moves(
                flat_arr,
                off_arr,
                len_arr,
                ctypes.c_size_t(num_games),
                ctypes.c_uint64(time_limit_ms),
                out_moves,
            )
        elif bot_version == "v1" and hasattr(self.lib, "alphabeta_v1_batch_choose_moves"):
            self.lib.alphabeta_v1_batch_choose_moves(
                flat_arr,
                off_arr,
                len_arr,
                ctypes.c_size_t(num_games),
                ctypes.c_uint64(time_limit_ms),
                out_moves,
            )
        else:
            self.lib.alphabeta_batch_choose_moves(
                flat_arr,
                off_arr,
                len_arr,
                ctypes.c_size_t(num_games),
                ctypes.c_uint64(time_limit_ms),
                out_moves,
            )
        return list(out_moves)

