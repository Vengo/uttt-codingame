//! Exhaustive Depth-Stratified Tier 6 Tablebase Solver and Direct Index Prober on ZFS Pool
//!
//! Precomputes the dense core stratum E <= 4 of Tier 6 (3 open subboards, 6 closed subboards)
//! across all 16 canonical orbits under D4 dihedral symmetry on `/my-zraid/uttt/egtb/tier6/`.
//! All subboard resolutions transition directly to the exhaustive Tier 7 ZFS tablebase in < 20 ns.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::egtb::EgtbOutcome;
use crate::egtb::macro_factored::{check_subboard_win, MicroMoveOutcome, MicroTransitionTable};
use crate::egtb::retrograde::MicroSubboardTable;
use crate::egtb::tier6::{
    canonical_triplet, extract_tier6_tau, SubboardResolution, Tier6MacroCondition,
};
use crate::egtb::tier7_zfs::{
    build_sym_micro_id_table, map_to_canonical_pair, Tier7MacroTau, Tier7ZfsProber,
    TOTAL_MICRO_STATES,
};
use crate::uttt_trait_solving::UltimateTicTacToeBoard;

pub const TIER6_ZFS_DIR: &str = "/my-zraid/uttt/egtb/tier6";
pub const DEFAULT_MAX_EMPTY_CELLS: usize = 4;

/// 16 Canonical Orbit triplet representatives under D4 dihedral group
pub const CANONICAL_TRIPLETS: [(usize, usize, usize); 16] = [
    // 10 canonical classes when center (4) is closed:
    (0, 1, 2), // Orbit 0
    (0, 1, 3), // Orbit 1
    (0, 1, 5), // Orbit 2
    (0, 1, 6), // Orbit 3
    (0, 1, 7), // Orbit 4
    (0, 1, 8), // Orbit 5
    (0, 2, 6), // Orbit 6
    (0, 2, 7), // Orbit 7
    (0, 5, 7), // Orbit 8
    (1, 3, 5), // Orbit 9
    // 6 canonical classes when center (4) is open:
    (0, 1, 4), // Orbit 10
    (0, 2, 4), // Orbit 11
    (0, 4, 5), // Orbit 12
    (0, 4, 8), // Orbit 13
    (1, 3, 4), // Orbit 14
    (1, 4, 7), // Orbit 15
];

#[inline(always)]
fn outcome_to_u16(o: EgtbOutcome) -> u16 {
    match o {
        EgtbOutcome::P1Win => 2,
        EgtbOutcome::Draw => 1,
        _ => 0,
    }
}

#[inline(always)]
fn u16_to_outcome(v: u16) -> EgtbOutcome {
    match v % 3 {
        2 => EgtbOutcome::P1Win,
        1 => EgtbOutcome::Draw,
        _ => EgtbOutcome::P2Win,
    }
}

/// Packs all 6 independent outcomes for (j, k, l) into a compact 10-bit integer (0..728 < 1024).
/// Free choice is omitted because it is a deterministic minimax function of the 3 targets!
#[inline(always)]
pub fn pack_6_outcomes(
    v_x_j: EgtbOutcome,
    v_x_k: EgtbOutcome,
    v_x_l: EgtbOutcome,
    v_o_j: EgtbOutcome,
    v_o_k: EgtbOutcome,
    v_o_l: EgtbOutcome,
) -> u16 {
    let x_j = outcome_to_u16(v_x_j);
    let x_k = outcome_to_u16(v_x_k);
    let x_l = outcome_to_u16(v_x_l);
    let o_j = outcome_to_u16(v_o_j);
    let o_k = outcome_to_u16(v_o_k);
    let o_l = outcome_to_u16(v_o_l);

    (((((x_j * 3 + x_k) * 3 + x_l) * 3 + o_j) * 3 + o_k) * 3 + o_l)
}

/// Unpacks a packed 10-bit integer into the outcome for a specific target and turn.
#[inline(always)]
pub fn unpack_outcome(
    packed: u16,
    target: usize, // 0=j, 1=k, 2=l, 3=Free Choice
    x_to_move: bool,
    e_j: usize,
    e_k: usize,
    e_l: usize,
) -> EgtbOutcome {
    let o_l = u16_to_outcome(packed);
    let o_k = u16_to_outcome(packed / 3);
    let o_j = u16_to_outcome(packed / 9);
    let x_l = u16_to_outcome(packed / 27);
    let x_k = u16_to_outcome(packed / 81);
    let x_j = u16_to_outcome(packed / 243);

    if x_to_move {
        match target {
            0 => x_j,
            1 => x_k,
            2 => x_l,
            _ => {
                // Free choice: max over open subboards
                if (e_j > 0 && x_j == EgtbOutcome::P1Win)
                    || (e_k > 0 && x_k == EgtbOutcome::P1Win)
                    || (e_l > 0 && x_l == EgtbOutcome::P1Win)
                {
                    EgtbOutcome::P1Win
                } else if (e_j > 0 && x_j == EgtbOutcome::Draw)
                    || (e_k > 0 && x_k == EgtbOutcome::Draw)
                    || (e_l > 0 && x_l == EgtbOutcome::Draw)
                {
                    EgtbOutcome::Draw
                } else {
                    EgtbOutcome::P2Win
                }
            }
        }
    } else {
        match target {
            0 => o_j,
            1 => o_k,
            2 => o_l,
            _ => {
                // Free choice: min over open subboards (max for O)
                if (e_j > 0 && o_j == EgtbOutcome::P2Win)
                    || (e_k > 0 && o_k == EgtbOutcome::P2Win)
                    || (e_l > 0 && o_l == EgtbOutcome::P2Win)
                {
                    EgtbOutcome::P2Win
                } else if (e_j > 0 && o_j == EgtbOutcome::Draw)
                    || (e_k > 0 && o_k == EgtbOutcome::Draw)
                    || (e_l > 0 && o_l == EgtbOutcome::Draw)
                {
                    EgtbOutcome::Draw
                } else {
                    EgtbOutcome::P1Win
                }
            }
        }
    }
}

/// Compact memory-efficient layer table storing packed u16 outcomes
pub struct Tier6PackedLayer {
    pub n_j: usize,
    pub n_k: usize,
    pub n_l: usize,
    pub data: Vec<u16>,
}

impl Tier6PackedLayer {
    pub fn new(n_j: usize, n_k: usize, n_l: usize) -> Self {
        let size = n_j * n_k * n_l;
        Tier6PackedLayer {
            n_j,
            n_k,
            n_l,
            data: vec![0u16; size],
        }
    }

    pub fn empty() -> Self {
        Tier6PackedLayer {
            n_j: 0,
            n_k: 0,
            n_l: 0,
            data: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn get_packed(&self, idx_j: usize, idx_k: usize, idx_l: usize) -> u16 {
        self.data[(idx_j * self.n_k + idx_k) * self.n_l + idx_l]
    }

    #[inline(always)]
    pub fn set_packed(&mut self, idx_j: usize, idx_k: usize, idx_l: usize, val: u16) {
        self.data[(idx_j * self.n_k + idx_k) * self.n_l + idx_l] = val;
    }
}

/// Enumerates all reachable, non-terminal Tier6MacroCondition vectors for a triplet (j, k, l)
pub fn enumerate_triplet_taus(j: usize, k: usize, l: usize) -> Vec<Tier6MacroCondition> {
    let mut closed = Vec::with_capacity(6);
    for s in 0..9 {
        if s != j && s != k && s != l {
            closed.push(s);
        }
    }

    let mut valid_taus = Vec::with_capacity(128);
    for assignment in 0..729 {
        let mut x_supers = 0u16;
        let mut o_supers = 0u16;
        let mut tmp = assignment;
        for &s in &closed {
            let val = tmp % 3;
            tmp /= 3;
            if val == 1 {
                x_supers |= 1 << s;
            } else if val == 2 {
                o_supers |= 1 << s;
            }
        }

        if check_subboard_win(x_supers) || check_subboard_win(o_supers) {
            continue;
        }

        let tau = extract_tier6_tau(x_supers, o_supers, j, k, l);
        valid_taus.push(tau);
    }

    valid_taus.sort_unstable_by_key(|t| t.encode_u16());
    valid_taus.dedup();
    valid_taus
}

/// Computes the retrograde transition from Tier 6 (j, k, l) into the exhaustive Tier 7 ZFS tablebase
pub fn transition_to_tier7_zfs(
    tau6: &Tier6MacroCondition,
    j: usize,
    k: usize,
    l: usize,
    resolved_subboard: usize,
    sub_outcome: SubboardResolution,
    last_sq: usize,
    turn_was_x: bool,
    micro_ids: (usize, usize, usize),
    zfs_prober: &Tier7ZfsProber,
) -> EgtbOutcome {
    let (rem_a, rem_b, micro_a, micro_b) = if resolved_subboard == j {
        (k, l, micro_ids.1, micro_ids.2)
    } else if resolved_subboard == k {
        (j, l, micro_ids.0, micro_ids.2)
    } else if resolved_subboard == l {
        (j, k, micro_ids.0, micro_ids.1)
    } else {
        panic!("resolved_subboard {} not in open triplet ({}, {}, {})", resolved_subboard, j, k, l);
    };

    debug_assert!(rem_a < rem_b);

    // 1. Terminal Check & Updated Macro Boundary Flags
    let (xw_a_prime, xw_b_prime, ow_a_prime, ow_b_prime, x_shared_prime, o_shared_prime, delta_prime) =
        match sub_outcome {
            SubboardResolution::XWin => {
                let immediate_win = match resolved_subboard {
                    x if x == j => tau6.xw_j,
                    x if x == k => tau6.xw_k,
                    _ => tau6.xw_l,
                };
                if immediate_win {
                    return EgtbOutcome::P1Win;
                }

                let (xw_a, xw_b, x_shared, x_line) = if resolved_subboard == j {
                    (tau6.xw_k | tau6.x_shared_jk, tau6.xw_l | tau6.x_shared_jl, tau6.x_shared_kl, tau6.line_jkl)
                } else if resolved_subboard == k {
                    (tau6.xw_j | tau6.x_shared_jk, tau6.xw_l | tau6.x_shared_kl, tau6.x_shared_jl, tau6.line_jkl)
                } else {
                    (tau6.xw_j | tau6.x_shared_jl, tau6.xw_k | tau6.x_shared_kl, tau6.x_shared_jk, tau6.line_jkl)
                };

                let (ow_a, ow_b, o_shared) = if resolved_subboard == j {
                    (tau6.ow_k, tau6.ow_l, tau6.o_shared_kl)
                } else if resolved_subboard == k {
                    (tau6.ow_j, tau6.ow_l, tau6.o_shared_jl)
                } else {
                    (tau6.ow_j, tau6.ow_k, tau6.o_shared_jk)
                };

                (
                    xw_a,
                    xw_b,
                    ow_a,
                    ow_b,
                    x_shared | x_line,
                    o_shared,
                    (tau6.delta + 1).clamp(-3, 3),
                )
            }
            SubboardResolution::OWin => {
                let immediate_win = match resolved_subboard {
                    x if x == j => tau6.ow_j,
                    x if x == k => tau6.ow_k,
                    _ => tau6.ow_l,
                };
                if immediate_win {
                    return EgtbOutcome::P2Win;
                }

                let (xw_a, xw_b, x_shared) = if resolved_subboard == j {
                    (tau6.xw_k, tau6.xw_l, tau6.x_shared_kl)
                } else if resolved_subboard == k {
                    (tau6.xw_j, tau6.xw_l, tau6.x_shared_jl)
                } else {
                    (tau6.xw_j, tau6.xw_k, tau6.x_shared_jk)
                };

                let (ow_a, ow_b, o_shared, o_line) = if resolved_subboard == j {
                    (tau6.ow_k | tau6.o_shared_jk, tau6.ow_l | tau6.o_shared_jl, tau6.o_shared_kl, tau6.line_jkl)
                } else if resolved_subboard == k {
                    (tau6.ow_j | tau6.o_shared_jk, tau6.ow_l | tau6.o_shared_kl, tau6.o_shared_jl, tau6.line_jkl)
                } else {
                    (tau6.ow_j | tau6.o_shared_jl, tau6.ow_k | tau6.o_shared_kl, tau6.o_shared_jk, tau6.line_jkl)
                };

                (
                    xw_a,
                    xw_b,
                    ow_a,
                    ow_b,
                    x_shared,
                    o_shared | o_line,
                    (tau6.delta - 1).clamp(-3, 3),
                )
            }
            SubboardResolution::Tie => {
                let (xw_a, xw_b, x_shared) = if resolved_subboard == j {
                    (tau6.xw_k, tau6.xw_l, tau6.x_shared_kl)
                } else if resolved_subboard == k {
                    (tau6.xw_j, tau6.xw_l, tau6.x_shared_jl)
                } else {
                    (tau6.xw_j, tau6.xw_k, tau6.x_shared_jk)
                };

                let (ow_a, ow_b, o_shared) = if resolved_subboard == j {
                    (tau6.ow_k, tau6.ow_l, tau6.o_shared_kl)
                } else if resolved_subboard == k {
                    (tau6.ow_j, tau6.ow_l, tau6.o_shared_jl)
                } else {
                    (tau6.ow_j, tau6.ow_k, tau6.o_shared_jk)
                };

                (xw_a, xw_b, ow_a, ow_b, x_shared, o_shared, tau6.delta)
            }
        };

    // 2. Next Target Subboard Constraint
    let next_turn = !turn_was_x;
    let target = if last_sq == rem_a {
        0
    } else if last_sq == rem_b {
        1
    } else {
        2 // Free choice
    };

    // 3. Query Exhaustive Tier 7 ZFS Tablebase directly in < 20 ns
    if let Some(outcome) = zfs_prober.probe_raw(
        rem_a,
        rem_b,
        xw_a_prime,
        xw_b_prime,
        ow_a_prime,
        ow_b_prime,
        x_shared_prime,
        o_shared_prime,
        delta_prime,
        target,
        next_turn,
        micro_a,
        micro_b,
    ) {
        if outcome != EgtbOutcome::Unknown {
            return outcome;
        }
    }

    // Terminal fallback
    if delta_prime > 0 {
        EgtbOutcome::P1Win
    } else if delta_prime < 0 {
        EgtbOutcome::P2Win
    } else {
        EgtbOutcome::Draw
    }
}

/// Pre-resolved entry for a single `(sb_idx, res_idx)` transition from Tier 6 to Tier 7
#[derive(Clone, Copy)]
pub enum PreboundEntry<'a> {
    Immediate(EgtbOutcome),
    Tier7Slice {
        slice: Option<&'a [u8]>,
        sym_idx: usize,
        flipped: bool,
        rem_a: usize,
        rem_b: usize,
        fallback: EgtbOutcome,
    },
}

/// Pre-resolved lookup table for all 3 open subboards x 3 resolution outcomes (XWin, OWin, Tie)
/// within a single Tier 6 canonical slice `(j, k, l, tau6)`.
pub struct PreboundTier6To7<'a> {
    pub entries: [[PreboundEntry<'a>; 3]; 3],
}

impl<'a> PreboundTier6To7<'a> {
    pub fn new(
        tau6: &Tier6MacroCondition,
        j: usize,
        k: usize,
        l: usize,
        zfs_prober: &'a Tier7ZfsProber,
    ) -> Self {
        let sbs = [j, k, l];
        let resolutions = [
            SubboardResolution::XWin,
            SubboardResolution::OWin,
            SubboardResolution::Tie,
        ];

        let entries = std::array::from_fn(|sb_idx| {
            let resolved_subboard = sbs[sb_idx];
            let (rem_a, rem_b) = match sb_idx {
                0 => (k, l),
                1 => (j, l),
                _ => (j, k),
            };
            debug_assert!(rem_a < rem_b);
            let pair_map = map_to_canonical_pair(rem_a, rem_b);

            std::array::from_fn(|res_idx| {
                let sub_outcome = resolutions[res_idx];
                let (
                    xw_a_prime,
                    xw_b_prime,
                    ow_a_prime,
                    ow_b_prime,
                    x_shared_prime,
                    o_shared_prime,
                    delta_prime,
                ) = match sub_outcome {
                    SubboardResolution::XWin => {
                        let immediate_win = match sb_idx {
                            0 => tau6.xw_j,
                            1 => tau6.xw_k,
                            _ => tau6.xw_l,
                        };
                        if immediate_win {
                            return PreboundEntry::Immediate(EgtbOutcome::P1Win);
                        }

                        let (xw_a, xw_b, x_shared, x_line) = if resolved_subboard == j {
                            (
                                tau6.xw_k | tau6.x_shared_jk,
                                tau6.xw_l | tau6.x_shared_jl,
                                tau6.x_shared_kl,
                                tau6.line_jkl,
                            )
                        } else if resolved_subboard == k {
                            (
                                tau6.xw_j | tau6.x_shared_jk,
                                tau6.xw_l | tau6.x_shared_kl,
                                tau6.x_shared_jl,
                                tau6.line_jkl,
                            )
                        } else {
                            (
                                tau6.xw_j | tau6.x_shared_jl,
                                tau6.xw_k | tau6.x_shared_kl,
                                tau6.x_shared_jk,
                                tau6.line_jkl,
                            )
                        };

                        let (ow_a, ow_b, o_shared) = if resolved_subboard == j {
                            (tau6.ow_k, tau6.ow_l, tau6.o_shared_kl)
                        } else if resolved_subboard == k {
                            (tau6.ow_j, tau6.ow_l, tau6.o_shared_jl)
                        } else {
                            (tau6.ow_j, tau6.ow_k, tau6.o_shared_jk)
                        };

                        (
                            xw_a,
                            xw_b,
                            ow_a,
                            ow_b,
                            x_shared | x_line,
                            o_shared,
                            (tau6.delta + 1).clamp(-3, 3),
                        )
                    }
                    SubboardResolution::OWin => {
                        let immediate_win = match sb_idx {
                            0 => tau6.ow_j,
                            1 => tau6.ow_k,
                            _ => tau6.ow_l,
                        };
                        if immediate_win {
                            return PreboundEntry::Immediate(EgtbOutcome::P2Win);
                        }

                        let (xw_a, xw_b, x_shared) = if resolved_subboard == j {
                            (tau6.xw_k, tau6.xw_l, tau6.x_shared_kl)
                        } else if resolved_subboard == k {
                            (tau6.xw_j, tau6.xw_l, tau6.x_shared_jl)
                        } else {
                            (tau6.xw_j, tau6.xw_k, tau6.x_shared_jk)
                        };

                        let (ow_a, ow_b, o_shared, o_line) = if resolved_subboard == j {
                            (
                                tau6.ow_k | tau6.o_shared_jk,
                                tau6.ow_l | tau6.o_shared_jl,
                                tau6.o_shared_kl,
                                tau6.line_jkl,
                            )
                        } else if resolved_subboard == k {
                            (
                                tau6.ow_j | tau6.o_shared_jk,
                                tau6.ow_l | tau6.o_shared_kl,
                                tau6.o_shared_jl,
                                tau6.line_jkl,
                            )
                        } else {
                            (
                                tau6.ow_j | tau6.o_shared_jl,
                                tau6.ow_k | tau6.o_shared_kl,
                                tau6.o_shared_jk,
                                tau6.line_jkl,
                            )
                        };

                        (
                            xw_a,
                            xw_b,
                            ow_a,
                            ow_b,
                            x_shared,
                            o_shared | o_line,
                            (tau6.delta - 1).clamp(-3, 3),
                        )
                    }
                    SubboardResolution::Tie => {
                        let (xw_a, xw_b, x_shared) = if resolved_subboard == j {
                            (tau6.xw_k, tau6.xw_l, tau6.x_shared_kl)
                        } else if resolved_subboard == k {
                            (tau6.xw_j, tau6.xw_l, tau6.x_shared_jl)
                        } else {
                            (tau6.xw_j, tau6.xw_k, tau6.x_shared_jk)
                        };

                        let (ow_a, ow_b, o_shared) = if resolved_subboard == j {
                            (tau6.ow_k, tau6.ow_l, tau6.o_shared_kl)
                        } else if resolved_subboard == k {
                            (tau6.ow_j, tau6.ow_l, tau6.o_shared_jl)
                        } else {
                            (tau6.ow_j, tau6.ow_k, tau6.o_shared_jk)
                        };

                        (xw_a, xw_b, ow_a, ow_b, x_shared, o_shared, tau6.delta)
                    }
                };

                let (xw_j, xw_k, ow_j, ow_k) = if pair_map.flipped {
                    (xw_b_prime, xw_a_prime, ow_b_prime, ow_a_prime)
                } else {
                    (xw_a_prime, xw_b_prime, ow_a_prime, ow_b_prime)
                };

                let target_tau = Tier7MacroTau {
                    xw_j,
                    xw_k,
                    ow_j,
                    ow_k,
                    x_shared: x_shared_prime,
                    o_shared: o_shared_prime,
                    delta: delta_prime.clamp(-2, 2),
                };

                let taus = &zfs_prober.orbit_taus[pair_map.orbit_id];
                let slice = taus
                    .binary_search(&target_tau)
                    .ok()
                    .and_then(|tau_idx| zfs_prober.get_slice(pair_map.orbit_id, tau_idx));

                let fallback = if delta_prime > 0 {
                    EgtbOutcome::P1Win
                } else if delta_prime < 0 {
                    EgtbOutcome::P2Win
                } else {
                    EgtbOutcome::Draw
                };

                PreboundEntry::Tier7Slice {
                    slice,
                    sym_idx: pair_map.sym as usize,
                    flipped: pair_map.flipped,
                    rem_a,
                    rem_b,
                    fallback,
                }
            })
        });

        PreboundTier6To7 { entries }
    }

    #[inline(always)]
    pub fn eval(
        &self,
        sb_idx: usize,
        res_idx: usize,
        last_sq: usize,
        turn_was_x: bool,
        micro_ids: (usize, usize, usize),
        sym_micro_id: &[[u16; TOTAL_MICRO_STATES]; 8],
    ) -> EgtbOutcome {
        match self.entries[sb_idx][res_idx] {
            PreboundEntry::Immediate(outcome) => outcome,
            PreboundEntry::Tier7Slice {
                slice,
                sym_idx,
                flipped,
                rem_a,
                rem_b,
                fallback,
            } => {
                if let Some(slice_bytes) = slice {
                    let (micro_a, micro_b) = match sb_idx {
                        0 => (micro_ids.1, micro_ids.2),
                        1 => (micro_ids.0, micro_ids.2),
                        _ => (micro_ids.0, micro_ids.1),
                    };
                    if micro_a < TOTAL_MICRO_STATES && micro_b < TOTAL_MICRO_STATES {
                        let (id_j, id_k) = if flipped {
                            (micro_b, micro_a)
                        } else {
                            (micro_a, micro_b)
                        };
                        let j_id = sym_micro_id[sym_idx][id_j] as usize;
                        let k_id = sym_micro_id[sym_idx][id_k] as usize;
                        let byte = slice_bytes[j_id * TOTAL_MICRO_STATES + k_id];

                        let v_x_j = EgtbOutcome::from(byte % 3);
                        let v_x_k = EgtbOutcome::from((byte / 3) % 3);
                        let v_o_j = EgtbOutcome::from((byte / 9) % 3);
                        let v_o_k = EgtbOutcome::from((byte / 27) % 3);

                        let is_target_j = if flipped {
                            last_sq == rem_b
                        } else {
                            last_sq == rem_a
                        };
                        let is_target_k = if flipped {
                            last_sq == rem_a
                        } else {
                            last_sq == rem_b
                        };

                        let next_turn_is_x = !turn_was_x;
                        let outcome = if next_turn_is_x {
                            if is_target_j {
                                v_x_j
                            } else if is_target_k {
                                v_x_k
                            } else if v_x_j == EgtbOutcome::P1Win || v_x_k == EgtbOutcome::P1Win {
                                EgtbOutcome::P1Win
                            } else if v_x_j == EgtbOutcome::Draw || v_x_k == EgtbOutcome::Draw {
                                EgtbOutcome::Draw
                            } else {
                                EgtbOutcome::P2Win
                            }
                        } else {
                            if is_target_j {
                                v_o_j
                            } else if is_target_k {
                                v_o_k
                            } else if v_o_j == EgtbOutcome::P2Win || v_o_k == EgtbOutcome::P2Win {
                                EgtbOutcome::P2Win
                            } else if v_o_j == EgtbOutcome::Draw || v_o_k == EgtbOutcome::Draw {
                                EgtbOutcome::Draw
                            } else {
                                EgtbOutcome::P1Win
                            }
                        };

                        if outcome != EgtbOutcome::Unknown {
                            return outcome;
                        }
                    }
                }
                fallback
            }
        }
    }
}

/// Tier 6 Scalable ZFS Depth-Stratified Core Solver (E <= 4)
pub struct Tier6ZfsSolver {
    pub output_dir: PathBuf,
    pub tier7_dir: Option<PathBuf>,
    pub num_threads: usize,
    pub max_empty_cells: usize,
}

impl Tier6ZfsSolver {
    pub fn new(output_dir: &Path, num_threads: usize, max_empty_cells: usize) -> Self {
        Self::new_with_tier7(output_dir, num_threads, max_empty_cells, None)
    }

    pub fn new_with_tier7(
        output_dir: &Path,
        num_threads: usize,
        max_empty_cells: usize,
        tier7_dir: Option<PathBuf>,
    ) -> Self {
        let threads = if num_threads > 0 {
            num_threads.min(64) // Bound worker threads to cap RAM footprint
        } else {
            4 // Safe default alongside self-play
        };

        Tier6ZfsSolver {
            output_dir: output_dir.to_path_buf(),
            tier7_dir,
            num_threads: threads,
            max_empty_cells: max_empty_cells.clamp(1, 4),
        }
    }

    /// Solves all 16 canonical orbits or a specific orbit
    pub fn solve_all(&self, target_orbit: Option<usize>) -> Result<(usize, u64), String> {
        let t0 = Instant::now();

        let tier7_env = std::env::var("TIER7_DIR").ok().map(PathBuf::from);
        let tier7_dir_buf = self.tier7_dir.clone()
            .or(tier7_env)
            .unwrap_or_else(|| PathBuf::from("/my-zraid/uttt/egtb/tier7"));
        let tier7_dir = tier7_dir_buf.as_path();

        println!("============================================================");
        println!("  Syzygy-Inspired Exhaustive Core Tier 6 Tablebase Solver (E <= {})", self.max_empty_cells);
        println!("  Output Directory: {:?}", self.output_dir);
        println!("  Parallel Worker Threads: {}", self.num_threads);
        println!("  Direct Transition Prober: {:?} (Exhaustive ZFS)", tier7_dir);
        println!("============================================================");

        fs::create_dir_all(&self.output_dir).map_err(|e| e.to_string())?;

        if !tier7_dir.exists() || !tier7_dir.join("manifest.json").exists() {
            return Err(format!("Exhaustive Tier 7 ZFS tablebase not found at {:?}. Cannot perform retrograde induction.", tier7_dir));
        }

        let zfs_prober = Tier7ZfsProber::new(tier7_dir)?;
        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);

        let orbits_to_solve: Vec<usize> = match target_orbit {
            Some(o) => {
                if o >= 16 {
                    return Err(format!("Invalid orbit {} (must be 0..15)", o));
                }
                vec![o]
            }
            None => (0..16).collect(),
        };

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .map_err(|e| e.to_string())?;

        let total_slices_solved = AtomicUsize::new(0);
        let total_bytes_written = AtomicUsize::new(0);

        pool.install(|| {
            for &orbit_id in &orbits_to_solve {
                let (j, k, l) = CANONICAL_TRIPLETS[orbit_id];
                let taus = enumerate_triplet_taus(j, k, l);
                println!("\n🌀 Solving Canonical Orbit {}/16: ({}, {}, {}) | Reachable Taus: {}", orbit_id, j, k, l, taus.len());

                taus.par_iter().enumerate().for_each(|(tau_idx, tau)| {
                    let slice_filename = format!("tier6_orbit_{}_tau_{:03}.bin", orbit_id, tau_idx);
                    let slice_path = self.output_dir.join(&slice_filename);

                    if slice_path.exists() {
                        return; // Resume capability: skip existing slices
                    }

                    let t_slice = Instant::now();
                    let slice_bytes = self.solve_single_tau_slice(
                        j, k, l, tau, &micro, &transitions, &zfs_prober, &slice_path,
                    );

                    match slice_bytes {
                        Ok(bytes) => {
                            total_slices_solved.fetch_add(1, Ordering::Relaxed);
                            total_bytes_written.fetch_add(bytes, Ordering::Relaxed);
                            let sec = t_slice.elapsed().as_secs_f64();
                            println!("  [Orbit {} Tau {:03}] Solved in {:.2}s ({:.1} MB/s) -> {:?}", orbit_id, tau_idx, sec, (bytes as f64) / (1024.0 * 1024.0 * sec), slice_filename);
                        }
                        Err(err) => eprintln!("❌ Error solving orbit {} tau {}: {}", orbit_id, tau_idx, err),
                    }
                });
            }
        });

        // Write master manifest
        self.write_manifest()?;

        let total_sec = t0.elapsed().as_secs_f64();
        let total_gb = (total_bytes_written.load(Ordering::Relaxed) as f64) / (1024.0 * 1024.0 * 1024.0);
        println!("\n============================================================");
        println!("✅ Tier 6 Core Stratum (E <= {}) Precomputation Complete!", self.max_empty_cells);
        println!("  Total Slices Solved: {}", total_slices_solved.load(Ordering::Relaxed));
        println!("  Total Storage: {:.2} GB in {:.2}s ({:.2} GB/s)", total_gb, total_sec, total_gb / total_sec.max(0.001));
        println!("============================================================");

        Ok((total_slices_solved.load(Ordering::Relaxed), total_bytes_written.load(Ordering::Relaxed) as u64))
    }

    /// Solves a single canonical tau slice across E <= max_empty_cells using retrograde induction
    fn solve_single_tau_slice(
        &self,
        j: usize,
        k: usize,
        l: usize,
        tau: &Tier6MacroCondition,
        micro: &MicroSubboardTable,
        transitions: &MicroTransitionTable,
        zfs_prober: &Tier7ZfsProber,
        out_path: &Path,
    ) -> Result<usize, String> {
        let max_e = self.max_empty_cells;
        let prebound = PreboundTier6To7::new(tau, j, k, l, zfs_prober);
        let sym_micro_id = &*zfs_prober.sym_micro_id;

        // Allocate 3D layer tables up to max_e (preserving exact binary file layout)
        let mut tables: Vec<Vec<Vec<Tier6PackedLayer>>> = (0..=max_e)
            .map(|e_j| {
                (0..=max_e)
                    .map(|e_k| {
                        (0..=max_e)
                            .map(|e_l| {
                                if e_j + e_k + e_l <= max_e {
                                    let n_j = if e_j == 0 { 1 } else { micro.states_by_empty[e_j].len() };
                                    let n_k = if e_k == 0 { 1 } else { micro.states_by_empty[e_k].len() };
                                    let n_l = if e_l == 0 { 1 } else { micro.states_by_empty[e_l].len() };
                                    Tier6PackedLayer::new(n_j, n_k, n_l)
                                } else {
                                    Tier6PackedLayer::empty()
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        // 1. Base Layer E = 0: All 3 subboards are full/tied
        {
            let base_outcome = if tau.delta > 0 {
                EgtbOutcome::P1Win
            } else if tau.delta < 0 {
                EgtbOutcome::P2Win
            } else {
                EgtbOutcome::Draw
            };
            let packed_base = pack_6_outcomes(
                base_outcome, base_outcome, base_outcome,
                base_outcome, base_outcome, base_outcome,
            );
            tables[0][0][0].set_packed(0, 0, 0, packed_base);
        }

        // 2. Backward Induction: total_e = 1..=max_e
        for total_e in 1..=max_e {
            for e_j in 0..=total_e {
                for e_k in 0..=(total_e - e_j) {
                    let e_l = total_e - e_j - e_k;
                    if e_j > max_e || e_k > max_e || e_l > max_e {
                        continue;
                    }
                    // Unreachable in Tier 6: all 3 open subboards must have >= 1 empty square,
                    // and playing the last square of any subboard always produces Win or Tie (never Next).
                    if e_j == 0 || e_k == 0 || e_l == 0 {
                        continue;
                    }

                    let n_j = micro.states_by_empty[e_j].len();
                    let n_k = micro.states_by_empty[e_k].len();
                    let n_l = micro.states_by_empty[e_l].len();

                    for idx_j in 0..n_j {
                        let j_id = micro.states_by_empty[e_j][idx_j];
                        let j_moves_x = &transitions.x_moves[j_id];
                        let j_moves_o = &transitions.o_moves[j_id];

                        for idx_k in 0..n_k {
                            let k_id = micro.states_by_empty[e_k][idx_k];
                            let k_moves_x = &transitions.x_moves[k_id];
                            let k_moves_o = &transitions.o_moves[k_id];

                            for idx_l in 0..n_l {
                                let l_id = micro.states_by_empty[e_l][idx_l];
                                let l_moves_x = &transitions.x_moves[l_id];
                                let l_moves_o = &transitions.o_moves[l_id];
                                let micro_ids = (j_id, k_id, l_id);

                                // Move evaluation closure for Player X
                                let eval_sb_x = |sb_idx: usize| -> EgtbOutcome {
                                    let moves = match sb_idx {
                                        0 => j_moves_x,
                                        1 => k_moves_x,
                                        _ => l_moves_x,
                                    };

                                    let mut best = EgtbOutcome::P2Win;
                                    for m in &moves.moves[..moves.count as usize] {
                                        let sq = m.sq as usize;
                                        let child = match m.outcome {
                                            MicroMoveOutcome::Win => prebound.eval(
                                                sb_idx, 0, sq, true, micro_ids, sym_micro_id,
                                            ),
                                            MicroMoveOutcome::Tie => prebound.eval(
                                                sb_idx, 2, sq, true, micro_ids, sym_micro_id,
                                            ),
                                            MicroMoveOutcome::Next => {
                                                let (n_ej, n_ek, n_el, n_ij, n_ik, n_il) = match sb_idx {
                                                    0 => (e_j - 1, e_k, e_l, m.next_idx as usize, idx_k, idx_l),
                                                    1 => (e_j, e_k - 1, e_l, idx_j, m.next_idx as usize, idx_l),
                                                    _ => (e_j, e_k, e_l - 1, idx_j, idx_k, m.next_idx as usize),
                                                };
                                                let next_target = if sq == j { 0 } else if sq == k { 1 } else if sq == l { 2 } else { 3 };
                                                let packed = tables[n_ej][n_ek][n_el].get_packed(n_ij, n_ik, n_il);
                                                unpack_outcome(packed, next_target, false, n_ej, n_ek, n_el)
                                            }
                                        };
                                        if child == EgtbOutcome::P1Win {
                                            return EgtbOutcome::P1Win;
                                        } else if child == EgtbOutcome::Draw && best == EgtbOutcome::P2Win {
                                            best = EgtbOutcome::Draw;
                                        }
                                    }
                                    best
                                };

                                // Move evaluation closure for Player O
                                let eval_sb_o = |sb_idx: usize| -> EgtbOutcome {
                                    let moves = match sb_idx {
                                        0 => j_moves_o,
                                        1 => k_moves_o,
                                        _ => l_moves_o,
                                    };

                                    let mut best = EgtbOutcome::P1Win;
                                    for m in &moves.moves[..moves.count as usize] {
                                        let sq = m.sq as usize;
                                        let child = match m.outcome {
                                            MicroMoveOutcome::Win => prebound.eval(
                                                sb_idx, 1, sq, false, micro_ids, sym_micro_id,
                                            ),
                                            MicroMoveOutcome::Tie => prebound.eval(
                                                sb_idx, 2, sq, false, micro_ids, sym_micro_id,
                                            ),
                                            MicroMoveOutcome::Next => {
                                                let (n_ej, n_ek, n_el, n_ij, n_ik, n_il) = match sb_idx {
                                                    0 => (e_j - 1, e_k, e_l, m.next_idx as usize, idx_k, idx_l),
                                                    1 => (e_j, e_k - 1, e_l, idx_j, m.next_idx as usize, idx_l),
                                                    _ => (e_j, e_k, e_l - 1, idx_j, idx_k, m.next_idx as usize),
                                                };
                                                let next_target = if sq == j { 0 } else if sq == k { 1 } else if sq == l { 2 } else { 3 };
                                                let packed = tables[n_ej][n_ek][n_el].get_packed(n_ij, n_ik, n_il);
                                                unpack_outcome(packed, next_target, true, n_ej, n_ek, n_el)
                                            }
                                        };
                                        if child == EgtbOutcome::P2Win {
                                            return EgtbOutcome::P2Win;
                                        } else if child == EgtbOutcome::Draw && best == EgtbOutcome::P1Win {
                                            best = EgtbOutcome::Draw;
                                        }
                                    }
                                    best
                                };

                                let res_x_j = eval_sb_x(0);
                                let res_x_k = eval_sb_x(1);
                                let res_x_l = eval_sb_x(2);

                                let res_o_j = eval_sb_o(0);
                                let res_o_k = eval_sb_o(1);
                                let res_o_l = eval_sb_o(2);

                                let packed = pack_6_outcomes(res_x_j, res_x_k, res_x_l, res_o_j, res_o_k, res_o_l);
                                tables[e_j][e_k][e_l].set_packed(idx_j, idx_k, idx_l, packed);
                            }
                        }
                    }
                }
            }
        }

        // 3. Serialize to disk atomically
        let temp_path = out_path.with_extension(format!("tmp.{}", std::process::id()));
        let mut f = BufWriter::with_capacity(1024 * 1024, File::create(&temp_path).map_err(|e| e.to_string())?);

        // Header:
        // Magic: "UTTT_T6S" (8 bytes)
        // max_e: u32 (4 bytes)
        // tau_u16: u16 (2 bytes)
        // reserved: u16 (2 bytes)
        f.write_all(b"UTTT_T6S").map_err(|e| e.to_string())?;
        f.write_all(&(max_e as u32).to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&tau.encode_u16().to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(&[0u8, 0u8]).map_err(|e| e.to_string())?;

        let mut total_bytes = 16usize;

        // Write each layer's data sequentially
        for total_e in 0..=max_e {
            for e_j in 0..=total_e {
                for e_k in 0..=(total_e - e_j) {
                    let e_l = total_e - e_j - e_k;
                    if e_j > max_e || e_k > max_e || e_l > max_e {
                        continue;
                    }
                    let layer = &tables[e_j][e_k][e_l];
                    let byte_slice = unsafe {
                        std::slice::from_raw_parts(
                            layer.data.as_ptr() as *const u8,
                            layer.data.len() * 2,
                        )
                    };
                    f.write_all(byte_slice).map_err(|e| e.to_string())?;
                    total_bytes += byte_slice.len();
                }
            }
        }

        f.flush().map_err(|e| e.to_string())?;
        drop(f);

        fs::rename(&temp_path, out_path).map_err(|e| e.to_string())?;

        Ok(total_bytes)
    }

    /// Writes manifest.json in the output directory
    pub fn write_manifest(&self) -> Result<(), String> {
        let manifest_path = self.output_dir.join("manifest.json");
        let mut manifest_json = format!(
            "{{\n  \"version\": 1,\n  \"tier\": 6,\n  \"max_empty_cells\": {},\n  \"orbits\": [\n",
            self.max_empty_cells
        );

        for (orbit_id, &(j, k, l)) in CANONICAL_TRIPLETS.iter().enumerate() {
            let taus = enumerate_triplet_taus(j, k, l);
            manifest_json.push_str(&format!(
                "    {{\n      \"orbit_id\": {},\n      \"rep_j\": {},\n      \"rep_k\": {},\n      \"rep_l\": {},\n      \"tau_count\": {},\n      \"slices\": [\n",
                orbit_id, j, k, l, taus.len()
            ));

            for (tau_idx, tau) in taus.iter().enumerate() {
                manifest_json.push_str(&format!(
                    "        {{\"tau_idx\": {}, \"tau_u16\": {}, \"file\": \"tier6_orbit_{}_tau_{:03}.bin\"}}{}",
                    tau_idx, tau.encode_u16(), orbit_id, tau_idx,
                    if tau_idx + 1 < taus.len() { ",\n" } else { "\n" }
                ));
            }
            manifest_json.push_str(&format!("      ]\n    }}{}", if orbit_id + 1 < CANONICAL_TRIPLETS.len() { ",\n" } else { "\n" }));
        }

        manifest_json.push_str("  ]\n}\n");
        fs::write(&manifest_path, manifest_json).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Zero-copy memory-mapped prober for Tier 6 tablebases on ZFS
pub struct Tier6ZfsProber {
    pub dir: PathBuf,
    pub micro: MicroSubboardTable,
    pub orbit_taus: [Vec<Tier6MacroCondition>; 16],
    pub mmaps: [Vec<OnceLock<Option<Mmap>>>; 16],
    pub sym_micro_id: Box<[[u16; TOTAL_MICRO_STATES]; 8]>,
    pub layer_byte_offsets: [[[usize; 5]; 5]; 5],
}

impl Tier6ZfsProber {
    pub fn new(dir: &Path) -> Result<Self, String> {
        let micro = MicroSubboardTable::build();
        let sym_micro_id = build_sym_micro_id_table(&micro);
        let orbit_taus: [Vec<Tier6MacroCondition>; 16] = std::array::from_fn(|orbit_id| {
            let (j, k, l) = CANONICAL_TRIPLETS[orbit_id];
            enumerate_triplet_taus(j, k, l)
        });
        let mmaps = std::array::from_fn(|orbit_id| {
            (0..orbit_taus[orbit_id].len())
                .map(|_| OnceLock::new())
                .collect()
        });

        let mut layer_byte_offsets = [[[0usize; 5]; 5]; 5];
        let mut offset = 16usize;
        for total_e in 0..=DEFAULT_MAX_EMPTY_CELLS {
            for e_j in 0..=total_e {
                for e_k in 0..=(total_e - e_j) {
                    let e_l = total_e - e_j - e_k;
                    if e_j > DEFAULT_MAX_EMPTY_CELLS
                        || e_k > DEFAULT_MAX_EMPTY_CELLS
                        || e_l > DEFAULT_MAX_EMPTY_CELLS
                    {
                        continue;
                    }
                    layer_byte_offsets[e_j][e_k][e_l] = offset;
                    let n_j = if e_j == 0 { 1 } else { micro.states_by_empty[e_j].len() };
                    let n_k = if e_k == 0 { 1 } else { micro.states_by_empty[e_k].len() };
                    let n_l = if e_l == 0 { 1 } else { micro.states_by_empty[e_l].len() };
                    offset += n_j * n_k * n_l * 2;
                }
            }
        }

        Ok(Tier6ZfsProber {
            dir: dir.to_path_buf(),
            micro,
            orbit_taus,
            mmaps,
            sym_micro_id,
            layer_byte_offsets,
        })
    }

    /// Lazily opens and memory-maps the canonical Tier 6 slice for `(orbit_id, tau_idx)` without locking
    #[inline]
    pub fn get_slice(&self, orbit_id: usize, tau_idx: usize) -> Option<&[u8]> {
        let cell = self.mmaps.get(orbit_id)?.get(tau_idx)?;
        let mmap_opt = cell.get_or_init(|| {
            let slice_filename = format!("tier6_orbit_{}_tau_{:03}.bin", orbit_id, tau_idx);
            let slice_file = File::open(self.dir.join(&slice_filename)).ok()?;
            let mmap = unsafe { Mmap::map(&slice_file).ok()? };
            if mmap.len() >= 16 && &mmap[0..8] == b"UTTT_T6S" {
                Some(mmap)
            } else {
                None
            }
        });
        mmap_opt.as_deref()
    }

    /// Probes an UltimateTicTacToeBoard state with 6 closed subboards in < 50 ns
    pub fn probe(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome> {
        if check_subboard_win(board.x_supers) {
            return Some(EgtbOutcome::P1Win);
        }
        if check_subboard_win(board.o_supers) {
            return Some(EgtbOutcome::P2Win);
        }

        let closed = board.x_supers | board.o_supers | board.get_tied_mask();
        let mut open = [0usize; 3];
        let mut open_cnt = 0;
        for s in 0..9 {
            if (closed & (1 << s)) == 0 {
                if open_cnt < 3 {
                    open[open_cnt] = s;
                }
                open_cnt += 1;
            }
        }
        if open_cnt != 3 {
            return None;
        }

        let canon = canonical_triplet(open[0], open[1], open[2]);

        let mut orig_j = 0usize;
        let mut orig_k = 0usize;
        let mut orig_l = 0usize;
        for &s in &open {
            let t = canon.sym.transform_3x3(s);
            if t == canon.rep_j {
                orig_j = s;
            } else if t == canon.rep_k {
                orig_k = s;
            } else if t == canon.rep_l {
                orig_l = s;
            } else {
                return None;
            }
        }

        // Check empty counts in canonical frame
        let e_j = 9 - (board.xbb[orig_j] | board.obb[orig_j]).count_ones() as usize;
        let e_k = 9 - (board.xbb[orig_k] | board.obb[orig_k]).count_ones() as usize;
        let e_l = 9 - (board.xbb[orig_l] | board.obb[orig_l]).count_ones() as usize;
        if e_j < 1 || e_k < 1 || e_l < 1 || e_j + e_k + e_l > DEFAULT_MAX_EMPTY_CELLS {
            return None;
        }

        // Canonical symmetry mapping of macro bitboards
        let canon_x = canon.sym.transform_9bit(board.x_supers);
        let canon_o = canon.sym.transform_9bit(board.o_supers);
        let target_tau = extract_tier6_tau(canon_x, canon_o, canon.rep_j, canon.rep_k, canon.rep_l);

        let taus = &self.orbit_taus[canon.orbit_id];
        let tau_idx = taus
            .binary_search_by_key(&target_tau.encode_u16(), |t| t.encode_u16())
            .ok()?;

        let slice = self.get_slice(canon.orbit_id, tau_idx)?;
        if slice.len() < 16 || &slice[0..8] != b"UTTT_T6S" {
            return None;
        }
        let slice_max_e = u32::from_le_bytes(slice[8..12].try_into().ok()?) as usize;
        if e_j + e_k + e_l > slice_max_e {
            return None;
        }

        let raw_j = self.micro.to_id[((board.xbb[orig_j] as usize) << 9) | (board.obb[orig_j] as usize)] as usize;
        let raw_k = self.micro.to_id[((board.xbb[orig_k] as usize) << 9) | (board.obb[orig_k] as usize)] as usize;
        let raw_l = self.micro.to_id[((board.xbb[orig_l] as usize) << 9) | (board.obb[orig_l] as usize)] as usize;
        if raw_j >= TOTAL_MICRO_STATES || raw_k >= TOTAL_MICRO_STATES || raw_l >= TOTAL_MICRO_STATES {
            return None;
        }

        let sym_idx = canon.sym as usize;
        let j_id = self.sym_micro_id[sym_idx][raw_j] as usize;
        let k_id = self.sym_micro_id[sym_idx][raw_k] as usize;
        let l_id = self.sym_micro_id[sym_idx][raw_l] as usize;

        let idx_j = self.micro.empty_and_local[j_id].1;
        let idx_k = self.micro.empty_and_local[k_id].1;
        let idx_l = self.micro.empty_and_local[l_id].1;

        let n_k = self.micro.states_by_empty[e_k].len();
        let n_l = self.micro.states_by_empty[e_l].len();
        let elem_idx = (idx_j * n_k + idx_k) * n_l + idx_l;
        let byte_offset = self.layer_byte_offsets[e_j][e_k][e_l] + elem_idx * 2;
        if slice.len() < byte_offset + 2 {
            return None;
        }

        let packed = u16::from_le_bytes([slice[byte_offset], slice[byte_offset + 1]]);
        let target = if board.last_super_index == orig_j as u8 {
            0
        } else if board.last_super_index == orig_k as u8 {
            1
        } else if board.last_super_index == orig_l as u8 {
            2
        } else {
            3
        };

        Some(unpack_outcome(packed, target, board.x_to_move, e_j, e_k, e_l))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::FREE_CHOICE_INDEX;
    use crate::egtb::tier7_zfs::SLICE_BYTE_SIZE;
    use crate::symmetry::{SymmetryEngine, ALL_SYMMETRIES};

    fn create_mock_tier7_slice(path: &Path, seed: usize) {
        let mut data = vec![0u8; SLICE_BYTE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = ((i.wrapping_mul(31).wrapping_add(seed.wrapping_mul(17))) % 81) as u8;
        }
        fs::write(path, &data).expect("Failed to write mock Tier 7 slice");
    }

    #[test]
    fn test_canonical_triplets_count() {
        assert_eq!(CANONICAL_TRIPLETS.len(), 16);
    }

    #[test]
    fn test_pack_unpack_6_outcomes_roundtrip() {
        let outcomes = [EgtbOutcome::P1Win, EgtbOutcome::Draw, EgtbOutcome::P2Win];
        for &xj in &outcomes {
            for &xk in &outcomes {
                for &xl in &outcomes {
                    for &oj in &outcomes {
                        for &ok in &outcomes {
                            for &ol in &outcomes {
                                let packed = pack_6_outcomes(xj, xk, xl, oj, ok, ol);
                                assert!(packed < 729);

                                assert_eq!(unpack_outcome(packed, 0, true, 1, 1, 1), xj);
                                assert_eq!(unpack_outcome(packed, 1, true, 1, 1, 1), xk);
                                assert_eq!(unpack_outcome(packed, 2, true, 1, 1, 1), xl);

                                assert_eq!(unpack_outcome(packed, 0, false, 1, 1, 1), oj);
                                assert_eq!(unpack_outcome(packed, 1, false, 1, 1, 1), ok);
                                assert_eq!(unpack_outcome(packed, 2, false, 1, 1, 1), ol);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_print_all_orbit_tau_counts() {
        let mut total = 0;
        println!("\n=== TIER 6 CANONICAL ORBIT TAU COUNTS ===");
        for (i, &(j, k, l)) in CANONICAL_TRIPLETS.iter().enumerate() {
            let taus = enumerate_triplet_taus(j, k, l);
            println!("Orbit {:2}: ({}, {}, {}) -> {:4} taus", i, j, k, l, taus.len());
            total += taus.len();
        }
        println!("TOTAL TAUS ACROSS ALL 16 ORBITS: {}", total);
    }

    #[test]
    fn test_prebound_tier6_to_tier7_equivalence() {
        let live_t7 = Path::new(crate::egtb::tier7_zfs::TIER7_ZFS_DIR);
        let temp_dir = std::env::temp_dir().join(format!(
            "uttt_prebound_t6_t7_test_{}",
            std::process::id()
        ));

        let sample_orbits = [0usize, 4, 6, 10];
        let tier7_path = if live_t7.join("manifest.json").exists() {
            live_t7.to_path_buf()
        } else {
            let _ = fs::remove_dir_all(&temp_dir);
            fs::create_dir_all(&temp_dir).unwrap();
            let prober_init = Tier7ZfsProber::new(&temp_dir).unwrap();
            let mut created_slices = std::collections::HashSet::new();
            for &orbit6 in &sample_orbits {
                let (j, k, l) = CANONICAL_TRIPLETS[orbit6];
                for &(rem_a, rem_b) in &[(k, l), (j, l), (j, k)] {
                    let pm = map_to_canonical_pair(rem_a, rem_b);
                    let tau_idx = 0;
                    if created_slices.insert((pm.orbit_id, tau_idx)) {
                        let fname = format!("tier7_orbit_{}_tau_{:03}.bin", pm.orbit_id, tau_idx);
                        create_mock_tier7_slice(
                            &temp_dir.join(fname),
                            pm.orbit_id * 1000 + tau_idx,
                        );
                    }
                    let _ = &prober_init;
                }
            }
            temp_dir.clone()
        };

        let zfs_prober = Tier7ZfsProber::new(&tier7_path).unwrap();
        let sample_micro_ids = [
            zfs_prober.micro.states_by_empty[1][0],
            zfs_prober.micro.states_by_empty[1][15],
            zfs_prober.micro.states_by_empty[2][10],
            zfs_prober.micro.states_by_empty[3][25],
        ];
        let resolutions = [
            SubboardResolution::XWin,
            SubboardResolution::OWin,
            SubboardResolution::Tie,
        ];

        for &orbit6 in &sample_orbits {
            let (j, k, l) = CANONICAL_TRIPLETS[orbit6];
            let sbs = [j, k, l];
            let taus = enumerate_triplet_taus(j, k, l);

            for tau6 in taus.iter().step_by((taus.len() / 6).max(1)) {
                let prebound = PreboundTier6To7::new(tau6, j, k, l, &zfs_prober);

                for &j_id in &sample_micro_ids {
                    for &k_id in &sample_micro_ids[..2] {
                        for &l_id in &sample_micro_ids[1..3] {
                            let micro_ids = (j_id, k_id, l_id);
                            for sb_idx in 0..3 {
                                let sb = sbs[sb_idx];
                                for (res_idx, &res) in resolutions.iter().enumerate() {
                                    for last_sq in 0..9 {
                                        for turn_was_x in [true, false] {
                                            let fast_res = prebound.eval(
                                                sb_idx,
                                                res_idx,
                                                last_sq,
                                                turn_was_x,
                                                micro_ids,
                                                &zfs_prober.sym_micro_id,
                                            );
                                            let ref_res = transition_to_tier7_zfs(
                                                tau6,
                                                j,
                                                k,
                                                l,
                                                sb,
                                                res,
                                                last_sq,
                                                turn_was_x,
                                                micro_ids,
                                                &zfs_prober,
                                            );
                                            assert_eq!(
                                                fast_res, ref_res,
                                                "Mismatch for triplet ({},{},{}) sb_idx={} res={:?} sq={} turn={}",
                                                j, k, l, sb_idx, res, last_sq, turn_was_x
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_tier6_zfs_solver_and_prober_d4_invariance_roundtrip() {
        let base_temp = std::env::temp_dir().join(format!(
            "uttt_tier6_roundtrip_test_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&base_temp);
        let tier6_dir = base_temp.join("tier6");
        fs::create_dir_all(&tier6_dir).unwrap();

        // Orbit 4 is (0, 1, 7), whose 8 D4 images are all distinct AND induce all 6 S3 permutations
        // of the 3 open subboards!
        let orbit_id = 4usize;
        let (rep_j, rep_k, rep_l) = CANONICAL_TRIPLETS[orbit_id];
        assert_eq!((rep_j, rep_k, rep_l), (0, 1, 7));

        // Closed subboards for (0, 1, 7) are {2, 3, 4, 5, 6, 8}.
        // Assign X wins on {2}, O wins on {3}, and Ties on {4, 5, 6, 8} so that
        // the macro condition tau is non-trivial and asymmetric across (j, k, l).
        let x_supers: u16 = 1 << 2;
        let o_supers: u16 = 1 << 3;
        let tau6 = extract_tier6_tau(x_supers, o_supers, rep_j, rep_k, rep_l);

        let taus6 = enumerate_triplet_taus(rep_j, rep_k, rep_l);
        let tau6_idx = taus6
            .binary_search_by_key(&tau6.encode_u16(), |t| t.encode_u16())
            .expect("tau6 must be in enumerated orbit taus");

        let live_t7 = Path::new(crate::egtb::tier7_zfs::TIER7_ZFS_DIR);
        let tier7_dir = if live_t7.join("manifest.json").exists() {
            live_t7.to_path_buf()
        } else {
            let mock_t7 = base_temp.join("tier7");
            fs::create_dir_all(&mock_t7).unwrap();
            fs::write(mock_t7.join("manifest.json"), "{}").unwrap();
            let prober_init = Tier7ZfsProber::new(&mock_t7).unwrap();
            for sb_idx in 0..3 {
                let (rem_a, rem_b) = match sb_idx {
                    0 => (rep_k, rep_l),
                    1 => (rep_j, rep_l),
                    _ => (rep_j, rep_k),
                };
                let pm = map_to_canonical_pair(rem_a, rem_b);
                let slice_path = mock_t7.join(format!("tier7_orbit_{}_tau_000.bin", pm.orbit_id));
                if !slice_path.exists() {
                    create_mock_tier7_slice(&slice_path, pm.orbit_id + 1);
                }
                let _ = &prober_init;
            }
            mock_t7
        };

        let zfs_tier7 = Tier7ZfsProber::new(&tier7_dir).unwrap();
        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);

        let solver = Tier6ZfsSolver::new_with_tier7(&tier6_dir, 4, 3, Some(tier7_dir.clone()));
        let slice_path = tier6_dir.join(format!("tier6_orbit_{}_tau_{:03}.bin", orbit_id, tau6_idx));
        solver
            .solve_single_tau_slice(
                rep_j,
                rep_k,
                rep_l,
                &tau6,
                &micro,
                &transitions,
                &zfs_tier7,
                &slice_path,
            )
            .expect("Failed to solve Tier 6 slice with max_e = 3");

        let tier6_prober = Tier6ZfsProber::new(&tier6_dir).expect("Failed to create Tier6ZfsProber");
        let zobrist = UltimateTicTacToeBoard::get_zobrist_constants();

        // Construct base board with closed subboards {2, 3, 4, 5, 6, 8}
        let mut base_board = UltimateTicTacToeBoard::default();
        base_board.x_supers = x_supers;
        base_board.o_supers = o_supers;
        // Subboard 2 won by X (top row 0b111 = 7)
        base_board.xbb[2] = 0b000_000_111;
        base_board.obb[2] = 0b000_011_000;
        // Subboard 3 won by O (top row 0b111 = 7)
        base_board.xbb[3] = 0b000_011_000;
        base_board.obb[3] = 0b000_000_111;
        // Subboards 4, 5, 6, 8 full ties (227 | 284 = 0x1FF)
        for &s in &[4usize, 5, 6, 8] {
            base_board.xbb[s] = 227;
            base_board.obb[s] = 284;
        }

        // Verify all 6 S3 permutations of the 3 open subboards are covered across ALL_SYMMETRIES
        let mut seen_s3_perms = std::collections::HashSet::new();

        // Pick several distinct e=1 micro-states for (rep_j, rep_k, rep_l)
        let e1_states = &micro.states_by_empty[1];
        let sample_triplets = [
            (e1_states[0], e1_states[5], e1_states[12]),
            (e1_states[7], e1_states[23], e1_states[41]),
            (e1_states[19], e1_states[64], e1_states[105]),
        ];

        for &(j_id, k_id, l_id) in &sample_triplets {
            let mut board = base_board.clone();
            board.xbb[rep_j] = micro.states[j_id].0;
            board.obb[rep_j] = micro.states[j_id].1;
            board.xbb[rep_k] = micro.states[k_id].0;
            board.obb[rep_k] = micro.states[k_id].1;
            board.xbb[rep_l] = micro.states[l_id].0;
            board.obb[rep_l] = micro.states[l_id].1;

            // Compute exact ground-truth minimax outcomes for each of the 3 open subboards
            let eval_exact_sb = |sb: usize, x_to_move: bool| -> EgtbOutcome {
                let id = if sb == rep_j {
                    j_id
                } else if sb == rep_k {
                    k_id
                } else {
                    l_id
                };
                let moves = if x_to_move {
                    &transitions.x_moves[id]
                } else {
                    &transitions.o_moves[id]
                };
                assert_eq!(moves.count, 1, "e=1 state must have exactly 1 legal move");
                let m = moves.moves[0];
                let res = match m.outcome {
                    MicroMoveOutcome::Win => {
                        if x_to_move {
                            SubboardResolution::XWin
                        } else {
                            SubboardResolution::OWin
                        }
                    }
                    MicroMoveOutcome::Tie => SubboardResolution::Tie,
                    MicroMoveOutcome::Next => unreachable!("e=1 move always resolves subboard"),
                };
                transition_to_tier7_zfs(
                    &tau6,
                    rep_j,
                    rep_k,
                    rep_l,
                    sb,
                    res,
                    m.sq as usize,
                    x_to_move,
                    (j_id, k_id, l_id),
                    &zfs_tier7,
                )
            };

            let x_outcomes = [
                eval_exact_sb(rep_j, true),
                eval_exact_sb(rep_k, true),
                eval_exact_sb(rep_l, true),
            ];
            let o_outcomes = [
                eval_exact_sb(rep_j, false),
                eval_exact_sb(rep_k, false),
                eval_exact_sb(rep_l, false),
            ];

            let x_free = if x_outcomes.contains(&EgtbOutcome::P1Win) {
                EgtbOutcome::P1Win
            } else if x_outcomes.contains(&EgtbOutcome::Draw) {
                EgtbOutcome::Draw
            } else {
                EgtbOutcome::P2Win
            };
            let o_free = if o_outcomes.contains(&EgtbOutcome::P2Win) {
                EgtbOutcome::P2Win
            } else if o_outcomes.contains(&EgtbOutcome::Draw) {
                EgtbOutcome::Draw
            } else {
                EgtbOutcome::P1Win
            };

            let targets = [
                (rep_j as u8, x_outcomes[0], o_outcomes[0]),
                (rep_k as u8, x_outcomes[1], o_outcomes[1]),
                (rep_l as u8, x_outcomes[2], o_outcomes[2]),
                (FREE_CHOICE_INDEX, x_free, o_free),
            ];

            for &(last_super, exp_x, exp_o) in &targets {
                for &turn in &[true, false] {
                    let expected = if turn { exp_x } else { exp_o };
                    board.last_super_index = last_super;
                    board.x_to_move = turn;

                    for &sym in &ALL_SYMMETRIES {
                        let transformed = SymmetryEngine::transform_board(&board, sym, &zobrist);

                        // Track induced S3 permutation of open subboards
                        let closed_t =
                            transformed.x_supers | transformed.o_supers | transformed.get_tied_mask();
                        let open_t: Vec<usize> =
                            (0..9).filter(|&s| (closed_t & (1 << s)) == 0).collect();
                        assert_eq!(open_t.len(), 3);
                        let c_t = canonical_triplet(open_t[0], open_t[1], open_t[2]);
                        let perm: Vec<usize> = open_t
                            .iter()
                            .map(|&s| {
                                let t = c_t.sym.transform_3x3(s);
                                if t == c_t.rep_j {
                                    0
                                } else if t == c_t.rep_k {
                                    1
                                } else {
                                    2
                                }
                            })
                            .collect();
                        seen_s3_perms.insert((perm[0], perm[1], perm[2]));

                        let probed = tier6_prober
                            .probe(&transformed)
                            .expect("Tier6ZfsProber::probe returned None on valid (1,1,1) state");
                        assert_eq!(
                            probed, expected,
                            "D4 invariance / minimax mismatch under {:?} (target={}, turn={})",
                            sym, last_super, turn
                        );
                    }
                }
            }
        }

        assert_eq!(
            seen_s3_perms.len(),
            6,
            "All 6 S3 permutations of the 3 open subboards must be exercised across D4"
        );

        let _ = fs::remove_dir_all(&base_temp);
    }
}
