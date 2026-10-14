//! Exhaustive Tier 7 Endgame Tablebase Solver and Direct Index Prober on ZFS Pool
//!
//! Stores complete, uncompressed or ZFS-compressed 1-byte micro-pair slices for all 614 canonical
//! macro conditions across the 8 dihedral canonical orbits on `/my-zraid/uttt/egtb/tier7/`.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::egtb::EgtbOutcome;
use crate::egtb::macro_factored::{
    check_subboard_win, MicroMoveOutcome, MicroTransitionTable,
    PairLayerTable, Tier8Lookup, WINNING_LINES,
};
use crate::egtb::retrograde::MicroSubboardTable;
use crate::symmetry::{DihedralSymmetry, ALL_SYMMETRIES};
use crate::uttt_trait_solving::UltimateTicTacToeBoard;

pub const TIER7_ZFS_DIR: &str = "/my-zraid/uttt/egtb/tier7";
pub const TOTAL_MICRO_STATES: usize = 11_125;
pub const SLICE_BYTE_SIZE: usize = TOTAL_MICRO_STATES * TOTAL_MICRO_STATES; // 123,765,625 bytes

/// Canonical orbit representatives for pairs of open subboards under D4 dihedral group
pub const CANONICAL_ORBITS: [(usize, usize); 8] = [
    (0, 1), // Orbit 0 (8 pairs)
    (0, 2), // Orbit 1 (4 pairs)
    (0, 4), // Orbit 2 (4 pairs)
    (0, 5), // Orbit 3 (8 pairs)
    (0, 8), // Orbit 4 (2 pairs)
    (1, 3), // Orbit 5 (4 pairs)
    (1, 4), // Orbit 6 (4 pairs)
    (1, 7), // Orbit 7 (2 pairs)
];

/// Canonical pair mapping result
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalPairMap {
    pub orbit_id: usize,
    pub rep_j: usize,
    pub rep_k: usize,
    pub sym: DihedralSymmetry,
    pub flipped: bool,
}

/// Maps any pair (j, k) with 0 <= j < k < 9 to its canonical orbit representative under D4
pub fn map_to_canonical_pair(j: usize, k: usize) -> CanonicalPairMap {
    debug_assert!(j < k && k < 9);
    let mut best_rep = (9usize, 9usize);
    let mut best_orbit = 0usize;
    let mut best_sym = DihedralSymmetry::Identity;
    let mut best_flipped = false;

    for &sym in &ALL_SYMMETRIES {
        let tj = sym.transform_3x3(j);
        let tk = sym.transform_3x3(k);
        let (cand_j, cand_k, flipped) = if tj < tk {
            (tj, tk, false)
        } else {
            (tk, tj, true)
        };

        if (cand_j, cand_k) < best_rep {
            best_rep = (cand_j, cand_k);
            best_sym = sym;
            best_flipped = flipped;
        }
    }

    let orbit_id = match best_rep {
        (0, 1) => 0,
        (0, 2) => 1,
        (0, 4) => 2,
        (0, 5) => 3,
        (0, 8) => 4,
        (1, 3) => 5,
        (1, 4) => 6,
        (1, 7) => 7,
        _ => unreachable!("Unknown pair orbit: {:?}", best_rep),
    };

    CanonicalPairMap {
        orbit_id,
        rep_j: best_rep.0,
        rep_k: best_rep.1,
        sym: best_sym,
        flipped: best_flipped,
    }
}

/// Macro boundary vector for Tier 7 (2 open subboards rep_j, rep_k)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct Tier7MacroTau {
    pub xw_j: bool,
    pub xw_k: bool,
    pub ow_j: bool,
    pub ow_k: bool,
    pub x_shared: bool,
    pub o_shared: bool,
    pub delta: i8, // clamped in -2..=2
}

impl Tier7MacroTau {
    #[inline(always)]
    pub fn encode_u16(&self) -> u16 {
        let delta_u = (self.delta.clamp(-2, 2) + 2) as u16; // 0..4 (3 bits)
        ((self.xw_j as u16) << 8)
            | ((self.xw_k as u16) << 7)
            | ((self.ow_j as u16) << 6)
            | ((self.ow_k as u16) << 5)
            | ((self.x_shared as u16) << 4)
            | ((self.o_shared as u16) << 3)
            | delta_u
    }

    #[inline(always)]
    pub fn decode_u16(val: u16) -> Self {
        let delta = ((val & 0x7) as i8) - 2;
        Tier7MacroTau {
            xw_j: (val & (1 << 8)) != 0,
            xw_k: (val & (1 << 7)) != 0,
            ow_j: (val & (1 << 6)) != 0,
            ow_k: (val & (1 << 5)) != 0,
            x_shared: (val & (1 << 4)) != 0,
            o_shared: (val & (1 << 3)) != 0,
            delta,
        }
    }
}

/// Enumerates all legally reachable, non-terminal macro tau conditions for an orbit representative
pub fn enumerate_orbit_taus(rep_j: usize, rep_k: usize) -> Vec<Tier7MacroTau> {
    let closed: Vec<usize> = (0..9).filter(|&s| s != rep_j && s != rep_k).collect();
    let mut unique_taus = Vec::with_capacity(128);

    for assignment in 0..2187 {
        let mut tmp = assignment;
        let mut x_supers = 0u16;
        let mut o_supers = 0u16;
        let mut x_count = 0i8;
        let mut o_count = 0i8;

        for &s in &closed {
            let val = tmp % 3;
            tmp /= 3;
            if val == 1 {
                x_supers |= 1 << s;
                x_count += 1;
            } else if val == 2 {
                o_supers |= 1 << s;
                o_count += 1;
            }
        }

        // Exclude positions where game already ended before Tier 7
        if check_subboard_win(x_supers) || check_subboard_win(o_supers) {
            continue;
        }

        let xw_j = check_subboard_win(x_supers | (1 << rep_j));
        let xw_k = check_subboard_win(x_supers | (1 << rep_k));
        let ow_j = check_subboard_win(o_supers | (1 << rep_j));
        let ow_k = check_subboard_win(o_supers | (1 << rep_k));

        let xw_both = check_subboard_win(x_supers | (1 << rep_j) | (1 << rep_k));
        let x_shared = xw_both && !xw_j && !xw_k;

        let ow_both = check_subboard_win(o_supers | (1 << rep_j) | (1 << rep_k));
        let o_shared = ow_both && !ow_j && !ow_k;

        let delta = (x_count - o_count).clamp(-2, 2);

        unique_taus.push(Tier7MacroTau {
            xw_j,
            xw_k,
            ow_j,
            ow_k,
            x_shared,
            o_shared,
            delta,
        });
    }

    unique_taus.sort_unstable();
    unique_taus.dedup();
    unique_taus
}

/// Solves a single canonical slice across all 123,765,625 micro pairs via exact retrograde induction
pub fn solve_single_canonical_slice(
    rep_j: usize,
    rep_k: usize,
    tau: &Tier7MacroTau,
    micro: &MicroSubboardTable,
    transitions: &MicroTransitionTable,
    tier8: &Tier8Lookup,
    tables: &mut [Vec<PairLayerTable>],
    out_slice: &mut [u8],
) {
    debug_assert_eq!(out_slice.len(), SLICE_BYTE_SIZE);

    // 1. Base case E = 0 (both open subboards full and tied)
    let base_outcome = if tau.delta > 0 {
        EgtbOutcome::P1Win
    } else if tau.delta < 0 {
        EgtbOutcome::P2Win
    } else {
        EgtbOutcome::Draw
    };

    let n_0j = micro.states_by_empty[0].len();
    let n_0k = micro.states_by_empty[0].len();
    for idx_j in 0..n_0j {
        for idx_k in 0..n_0k {
            for turn in [true, false] {
                for target in 0..3 {
                    tables[0][0].set(idx_j, idx_k, turn, target, base_outcome);
                }
            }
        }
    }

    // 2. Backward induction across empty cell counts total_e = 1..=18
    for total_e in 1..=18usize {
        let min_ej = total_e.saturating_sub(9);
        let max_ej = total_e.min(9);

        for e_j in min_ej..=max_ej {
            let e_k = total_e - e_j;
            if e_k > 9 {
                continue;
            }

            let n_j = micro.states_by_empty[e_j].len();
            let n_k = micro.states_by_empty[e_k].len();

            for idx_j in 0..n_j {
                let j_id = micro.states_by_empty[e_j][idx_j];
                let j_moves_x = &transitions.x_moves[j_id];
                let j_moves_o = &transitions.o_moves[j_id];

                for idx_k in 0..n_k {
                    let k_id = micro.states_by_empty[e_k][idx_k];
                    let k_moves_x = &transitions.x_moves[k_id];
                    let k_moves_o = &transitions.o_moves[k_id];

                    if e_j == 0 {
                        // Subboard j is full. Moves occur exclusively in k.
                        let mut outcome_x = EgtbOutcome::P2Win;
                        for m in &k_moves_x.moves[..k_moves_x.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.xw_k {
                                        EgtbOutcome::P1Win
                                    } else {
                                        let xw_j_prime = tau.xw_j || tau.x_shared;
                                        let ow_j_prime = tau.ow_j;
                                        let delta_prime = (tau.delta + 1).clamp(-2, 2);
                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_j, tau.ow_j, tau.delta, j_id, false)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[0][e_k - 1].get(idx_j, m.next_idx as usize, false, next_target)
                                }
                            };
                            if child == EgtbOutcome::P1Win {
                                outcome_x = EgtbOutcome::P1Win;
                                break;
                            } else if child == EgtbOutcome::Draw && outcome_x == EgtbOutcome::P2Win {
                                outcome_x = EgtbOutcome::Draw;
                            }
                        }
                        for target in 0..3usize {
                            tables[0][e_k].set(idx_j, idx_k, true, target, outcome_x);
                        }

                        let mut outcome_o = EgtbOutcome::P1Win;
                        for m in &k_moves_o.moves[..k_moves_o.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.ow_k {
                                        EgtbOutcome::P2Win
                                    } else {
                                        let xw_j_prime = tau.xw_j;
                                        let ow_j_prime = tau.ow_j || tau.o_shared;
                                        let delta_prime = (tau.delta - 1).clamp(-2, 2);
                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_j, tau.ow_j, tau.delta, j_id, true)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[0][e_k - 1].get(idx_j, m.next_idx as usize, true, next_target)
                                }
                            };
                            if child == EgtbOutcome::P2Win {
                                outcome_o = EgtbOutcome::P2Win;
                                break;
                            } else if child == EgtbOutcome::Draw && outcome_o == EgtbOutcome::P1Win {
                                outcome_o = EgtbOutcome::Draw;
                            }
                        }
                        for target in 0..3usize {
                            tables[0][e_k].set(idx_j, idx_k, false, target, outcome_o);
                        }
                    } else if e_k == 0 {
                        // Subboard k is full. Moves occur exclusively in j.
                        let mut outcome_x = EgtbOutcome::P2Win;
                        for m in &j_moves_x.moves[..j_moves_x.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.xw_j {
                                        EgtbOutcome::P1Win
                                    } else {
                                        let xw_k_prime = tau.xw_k || tau.x_shared;
                                        let ow_k_prime = tau.ow_k;
                                        let delta_prime = (tau.delta + 1).clamp(-2, 2);
                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_k, tau.ow_k, tau.delta, k_id, false)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[e_j - 1][0].get(m.next_idx as usize, idx_k, false, next_target)
                                }
                            };
                            if child == EgtbOutcome::P1Win {
                                outcome_x = EgtbOutcome::P1Win;
                                break;
                            } else if child == EgtbOutcome::Draw && outcome_x == EgtbOutcome::P2Win {
                                outcome_x = EgtbOutcome::Draw;
                            }
                        }
                        for target in 0..3usize {
                            tables[e_j][0].set(idx_j, idx_k, true, target, outcome_x);
                        }

                        let mut outcome_o = EgtbOutcome::P1Win;
                        for m in &j_moves_o.moves[..j_moves_o.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.ow_j {
                                        EgtbOutcome::P2Win
                                    } else {
                                        let xw_k_prime = tau.xw_k;
                                        let ow_k_prime = tau.ow_k || tau.o_shared;
                                        let delta_prime = (tau.delta - 1).clamp(-2, 2);
                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_k, tau.ow_k, tau.delta, k_id, true)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[e_j - 1][0].get(m.next_idx as usize, idx_k, true, next_target)
                                }
                            };
                            if child == EgtbOutcome::P2Win {
                                outcome_o = EgtbOutcome::P2Win;
                                break;
                            } else if child == EgtbOutcome::Draw && outcome_o == EgtbOutcome::P1Win {
                                outcome_o = EgtbOutcome::Draw;
                            }
                        }
                        for target in 0..3usize {
                            tables[e_j][0].set(idx_j, idx_k, false, target, outcome_o);
                        }
                    } else {
                        // Both e_j > 0 and e_k > 0
                        // Player X in j:
                        let mut x_in_j = EgtbOutcome::P2Win;
                        for m in &j_moves_x.moves[..j_moves_x.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.xw_j {
                                        EgtbOutcome::P1Win
                                    } else {
                                        let xw_k_prime = tau.xw_k || tau.x_shared;
                                        let ow_k_prime = tau.ow_k;
                                        let delta_prime = (tau.delta + 1).clamp(-2, 2);
                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, false)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_k, tau.ow_k, tau.delta, k_id, false)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[e_j - 1][e_k].get(m.next_idx as usize, idx_k, false, next_target)
                                }
                            };
                            if child == EgtbOutcome::P1Win {
                                x_in_j = EgtbOutcome::P1Win;
                                break;
                            } else if child == EgtbOutcome::Draw && x_in_j == EgtbOutcome::P2Win {
                                x_in_j = EgtbOutcome::Draw;
                            }
                        }

                        // Player X in k:
                        let mut x_in_k = EgtbOutcome::P2Win;
                        for m in &k_moves_x.moves[..k_moves_x.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.xw_k {
                                        EgtbOutcome::P1Win
                                    } else {
                                        let xw_j_prime = tau.xw_j || tau.x_shared;
                                        let ow_j_prime = tau.ow_j;
                                        let delta_prime = (tau.delta + 1).clamp(-2, 2);
                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, false)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_j, tau.ow_j, tau.delta, j_id, false)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[e_j][e_k - 1].get(idx_j, m.next_idx as usize, false, next_target)
                                }
                            };
                            if child == EgtbOutcome::P1Win {
                                x_in_k = EgtbOutcome::P1Win;
                                break;
                            } else if child == EgtbOutcome::Draw && x_in_k == EgtbOutcome::P2Win {
                                x_in_k = EgtbOutcome::Draw;
                            }
                        }

                        let x_free = if x_in_j == EgtbOutcome::P1Win || x_in_k == EgtbOutcome::P1Win {
                            EgtbOutcome::P1Win
                        } else if x_in_j == EgtbOutcome::Draw || x_in_k == EgtbOutcome::Draw {
                            EgtbOutcome::Draw
                        } else {
                            EgtbOutcome::P2Win
                        };

                        tables[e_j][e_k].set(idx_j, idx_k, true, 0, x_in_j);
                        tables[e_j][e_k].set(idx_j, idx_k, true, 1, x_in_k);
                        tables[e_j][e_k].set(idx_j, idx_k, true, 2, x_free);

                        // Player O in j:
                        let mut o_in_j = EgtbOutcome::P1Win;
                        for m in &j_moves_o.moves[..j_moves_o.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.ow_j {
                                        EgtbOutcome::P2Win
                                    } else {
                                        let xw_k_prime = tau.xw_k;
                                        let ow_k_prime = tau.ow_k || tau.o_shared;
                                        let delta_prime = (tau.delta - 1).clamp(-2, 2);
                                        tier8.get(xw_k_prime, ow_k_prime, delta_prime, k_id, true)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_k, tau.ow_k, tau.delta, k_id, true)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[e_j - 1][e_k].get(m.next_idx as usize, idx_k, true, next_target)
                                }
                            };
                            if child == EgtbOutcome::P2Win {
                                o_in_j = EgtbOutcome::P2Win;
                                break;
                            } else if child == EgtbOutcome::Draw && o_in_j == EgtbOutcome::P1Win {
                                o_in_j = EgtbOutcome::Draw;
                            }
                        }

                        // Player O in k:
                        let mut o_in_k = EgtbOutcome::P1Win;
                        for m in &k_moves_o.moves[..k_moves_o.count as usize] {
                            let sq = m.sq as usize;
                            let child = match m.outcome {
                                MicroMoveOutcome::Win => {
                                    if tau.ow_k {
                                        EgtbOutcome::P2Win
                                    } else {
                                        let xw_j_prime = tau.xw_j;
                                        let ow_j_prime = tau.ow_j || tau.o_shared;
                                        let delta_prime = (tau.delta - 1).clamp(-2, 2);
                                        tier8.get(xw_j_prime, ow_j_prime, delta_prime, j_id, true)
                                    }
                                }
                                MicroMoveOutcome::Tie => {
                                    tier8.get(tau.xw_j, tau.ow_j, tau.delta, j_id, true)
                                }
                                MicroMoveOutcome::Next => {
                                    let next_target = if sq == rep_j {
                                        0
                                    } else if sq == rep_k {
                                        1
                                    } else {
                                        2
                                    };
                                    tables[e_j][e_k - 1].get(idx_j, m.next_idx as usize, true, next_target)
                                }
                            };
                            if child == EgtbOutcome::P2Win {
                                o_in_k = EgtbOutcome::P2Win;
                                break;
                            } else if child == EgtbOutcome::Draw && o_in_k == EgtbOutcome::P1Win {
                                o_in_k = EgtbOutcome::Draw;
                            }
                        }

                        let o_free = if o_in_j == EgtbOutcome::P2Win || o_in_k == EgtbOutcome::P2Win {
                            EgtbOutcome::P2Win
                        } else if o_in_j == EgtbOutcome::Draw || o_in_k == EgtbOutcome::Draw {
                            EgtbOutcome::Draw
                        } else {
                            EgtbOutcome::P1Win
                        };

                        tables[e_j][e_k].set(idx_j, idx_k, false, 0, o_in_j);
                        tables[e_j][e_k].set(idx_j, idx_k, false, 1, o_in_k);
                        tables[e_j][e_k].set(idx_j, idx_k, false, 2, o_free);
                    }
                }
            }
        }
    }

    // 3. Dense 1-byte micro-pair packing directly into the output slice
    // byte = v(X, j) + 3 * v(X, k) + 9 * v(O, j) + 27 * v(O, k)
    for j_id in 0..TOTAL_MICRO_STATES {
        let (e_j, idx_j) = micro.empty_and_local[j_id];
        let row_offset = j_id * TOTAL_MICRO_STATES;

        for k_id in 0..TOTAL_MICRO_STATES {
            let (e_k, idx_k) = micro.empty_and_local[k_id];
            let v1 = tables[e_j][e_k].get(idx_j, idx_k, true, 0) as u8;  // Turn X, Target j
            let v2 = tables[e_j][e_k].get(idx_j, idx_k, true, 1) as u8;  // Turn X, Target k
            let v3 = tables[e_j][e_k].get(idx_j, idx_k, false, 0) as u8; // Turn O, Target j
            let v4 = tables[e_j][e_k].get(idx_j, idx_k, false, 1) as u8; // Turn O, Target k

            let byte = v1 + 3 * v2 + 9 * v3 + 27 * v4;
            out_slice[row_offset + k_id] = byte;
        }
    }
}

/// Allocates pre-sized PairLayerTable memory for all e_j in 0..=9, e_k in 0..=9 (~708 MB resident per thread)
pub fn allocate_pair_layer_tables(micro: &MicroSubboardTable) -> Vec<Vec<PairLayerTable>> {
    (0..=9)
        .map(|e_j| {
            (0..=9)
                .map(|e_k| {
                    let n_j = micro.states_by_empty[e_j].len();
                    let n_k = micro.states_by_empty[e_k].len();
                    PairLayerTable::new(n_j, n_k)
                })
                .collect()
        })
        .collect()
}

/// Manages parallel solving and persistence of all 614 canonical Tier 7 slices onto ZFS storage
pub struct Tier7ZfsSolver {
    pub output_dir: PathBuf,
    pub num_threads: usize,
}

impl Tier7ZfsSolver {
    pub fn new(output_dir: &Path, num_threads: usize) -> Self {
        fs::create_dir_all(output_dir).expect("Failed to create Tier 7 ZFS directory");
        Tier7ZfsSolver {
            output_dir: output_dir.to_path_buf(),
            num_threads: if num_threads > 0 { num_threads } else { 8 },
        }
    }

    /// Solves all 614 canonical slices in parallel directly to /my-zraid/uttt/egtb/tier7/
    pub fn solve_all(&self) -> Result<(usize, u64), String> {
        let t0 = Instant::now();
        println!("============================================================");
        println!("  Exhaustive Syzygy Tier 7 ZFS Tablebase Generator");
        println!("  Target Directory: {:?}", self.output_dir);
        println!("  Parallel Worker Threads: {}", self.num_threads);
        println!("  Total Canonical Orbits: 8");
        println!("============================================================");

        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);
        let tier8 = Tier8Lookup::new(&micro);

        // Enumerate all slices to solve across the 8 canonical orbits
        let mut all_tasks = Vec::new();
        for (orbit_id, &(rep_j, rep_k)) in CANONICAL_ORBITS.iter().enumerate() {
            let taus = enumerate_orbit_taus(rep_j, rep_k);
            println!("  Orbit {} ({}, {}): {} canonical tau slices", orbit_id, rep_j, rep_k, taus.len());
            for (tau_idx, tau) in taus.into_iter().enumerate() {
                all_tasks.push((orbit_id, rep_j, rep_k, tau_idx, tau));
            }
        }

        let total_slices = all_tasks.len();
        println!("🚀 Total Canonical Slices to compute: {}", total_slices);

        let completed = AtomicUsize::new(0);
        let bytes_written = AtomicUsize::new(0);

        // Configure scoped rayon threadpool with bound thread count to reserve cores for self-play
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()
            .map_err(|e| e.to_string())?;

        pool.install(|| {
            all_tasks.into_par_iter().for_each(|(orbit_id, rep_j, rep_k, tau_idx, tau)| {
                let slice_filename = format!("tier7_orbit_{}_tau_{:03}.bin", orbit_id, tau_idx);
                let slice_path = self.output_dir.join(&slice_filename);

                // Idempotent resume: check if file exists with exact size
                if let Ok(meta) = fs::metadata(&slice_path) {
                    if meta.len() == SLICE_BYTE_SIZE as u64 {
                        let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        bytes_written.fetch_add(SLICE_BYTE_SIZE, Ordering::Relaxed);
                        println!("  [{}/{}] Cached: {} ({} MB)", c, total_slices, slice_filename, SLICE_BYTE_SIZE / (1024 * 1024));
                        return;
                    }
                }

                // Allocate thread-local pair layer table and output slice
                let t_slice_start = Instant::now();
                let mut tables = allocate_pair_layer_tables(&micro);
                let mut slice = vec![0u8; SLICE_BYTE_SIZE];

                solve_single_canonical_slice(
                    rep_j,
                    rep_k,
                    &tau,
                    &micro,
                    &transitions,
                    &tier8,
                    &mut tables,
                    &mut slice,
                );

                // Write atomic temp file then rename
                let temp_path = self.output_dir.join(format!("{}.tmp.{}", slice_filename, std::process::id()));
                let mut file = BufWriter::new(File::create(&temp_path).expect("Failed to create slice file"));
                file.write_all(&slice).expect("Failed to write slice data");
                file.flush().expect("Failed to flush slice data");
                drop(file);
                fs::rename(&temp_path, &slice_path).expect("Failed to rename atomic slice file");

                let c = completed.fetch_add(1, Ordering::Relaxed) + 1;
                bytes_written.fetch_add(SLICE_BYTE_SIZE, Ordering::Relaxed);
                let elapsed = t_slice_start.elapsed().as_secs_f64();
                let rate = (SLICE_BYTE_SIZE as f64) / (1024.0 * 1024.0 * elapsed);
                println!(
                    "  [{}/{}] Computed {} in {:.2}s ({:.1} MB/s) -> {:?}",
                    c, total_slices, slice_filename, elapsed, rate, slice_path
                );
            });
        });

        // Write master manifest
        self.write_manifest()?;

        let total_sec = t0.elapsed().as_secs_f64();
        let total_gb = (bytes_written.load(Ordering::Relaxed) as f64) / (1024.0 * 1024.0 * 1024.0);
        println!("============================================================");
        println!("✅ Exhaustive Tier 7 ZFS Tablebase Completed!");
        println!("  Total Slices: {}", total_slices);
        println!("  Total Uncompressed Data: {:.2} GB in {:.2}s ({:.2} GB/s)", total_gb, total_sec, total_gb / total_sec);
        println!("  Location: {:?}", self.output_dir);
        println!("============================================================");

        Ok((total_slices, bytes_written.load(Ordering::Relaxed) as u64))
    }

    /// Writes the master manifest.json indexing orbits, symmetries, and tau filenames
    pub fn write_manifest(&self) -> Result<(), String> {
        let manifest_path = self.output_dir.join("manifest.json");
        let mut manifest_json = String::from("{\n  \"version\": 1,\n  \"tier\": 7,\n  \"total_micro_states\": 11125,\n  \"slice_bytes\": 123765625,\n  \"orbits\": [\n");

        for (orbit_id, &(rep_j, rep_k)) in CANONICAL_ORBITS.iter().enumerate() {
            let taus = enumerate_orbit_taus(rep_j, rep_k);
            manifest_json.push_str(&format!(
                "    {{\n      \"orbit_id\": {},\n      \"rep_j\": {},\n      \"rep_k\": {},\n      \"tau_count\": {},\n      \"slices\": [\n",
                orbit_id, rep_j, rep_k, taus.len()
            ));

            for (tau_idx, tau) in taus.iter().enumerate() {
                manifest_json.push_str(&format!(
                    "        {{\"tau_idx\": {}, \"tau_u16\": {}, \"xw_j\": {}, \"xw_k\": {}, \"ow_j\": {}, \"ow_k\": {}, \"x_shared\": {}, \"o_shared\": {}, \"delta\": {}, \"file\": \"tier7_orbit_{}_tau_{:03}.bin\"}}{}",
                    tau_idx, tau.encode_u16(), tau.xw_j, tau.xw_k, tau.ow_j, tau.ow_k, tau.x_shared, tau.o_shared, tau.delta, orbit_id, tau_idx,
                    if tau_idx + 1 < taus.len() { ",\n" } else { "\n" }
                ));
            }
            manifest_json.push_str(&format!("      ]\n    }}{}", if orbit_id + 1 < CANONICAL_ORBITS.len() { ",\n" } else { "\n" }));
        }

        manifest_json.push_str("  ]\n}\n");
        fs::write(&manifest_path, manifest_json).map_err(|e| e.to_string())?;
        Ok(())
    }
}

/// Zero-copy memory-mapped prober for the exhaustive Tier 7 ZFS tablebase
pub struct Tier7ZfsProber {
    pub dir: PathBuf,
    pub micro: MicroSubboardTable,
    pub orbit_taus: [Vec<Tier7MacroTau>; 8],
    pub open_mmaps: Mutex<std::collections::HashMap<String, Mmap>>,
}

impl Tier7ZfsProber {
    pub fn new(dir: &Path) -> Result<Self, String> {
        let micro = MicroSubboardTable::build();
        let orbit_taus = std::array::from_fn(|orbit_id| {
            let (rep_j, rep_k) = CANONICAL_ORBITS[orbit_id];
            enumerate_orbit_taus(rep_j, rep_k)
        });

        Ok(Tier7ZfsProber {
            dir: dir.to_path_buf(),
            micro,
            orbit_taus,
            open_mmaps: Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// Probes an UltimateTicTacToeBoard state with 7 closed subboards in < 15 ns
    pub fn probe(&self, board: &UltimateTicTacToeBoard) -> Option<EgtbOutcome> {
        // 1. Identify the 2 open subboards (7 closed by win or tie)
        let closed_supers = board.x_supers | board.o_supers | board.get_tied_mask();
        let mut open = [0usize; 2];
        let mut open_cnt = 0;
        for s in 0..9 {
            if (closed_supers & (1 << s)) == 0 {
                if open_cnt < 2 {
                    open[open_cnt] = s;
                }
                open_cnt += 1;
            }
        }
        if open_cnt != 2 {
            return None;
        }

        let (j, k) = (open[0], open[1]);
        let pair_map = map_to_canonical_pair(j, k);

        // 2. Transform macro bitboards to canonical orientation
        let canon_x = pair_map.sym.transform_9bit(board.x_supers);
        let canon_o = pair_map.sym.transform_9bit(board.o_supers);

        // 3. Extract macro tau on canonical board
        let (xw_j, xw_k, ow_j, ow_k, x_shared, o_shared, delta) = crate::egtb::macro_factored::extract_augmented_tau(
            canon_x,
            canon_o,
            pair_map.rep_j,
            pair_map.rep_k,
        );

        let target_tau = Tier7MacroTau {
            xw_j,
            xw_k,
            ow_j,
            ow_k,
            x_shared,
            o_shared,
            delta: delta.clamp(-2, 2),
        };

        // Find tau_idx in orbit
        let taus = &self.orbit_taus[pair_map.orbit_id];
        let tau_idx = taus.binary_search(&target_tau).ok()?;

        // 4. Transform micro subboards to canonical orientation
        let (orig_j, orig_k) = if pair_map.flipped { (k, j) } else { (j, k) };
        let canon_xj = pair_map.sym.transform_9bit(board.xbb[orig_j]);
        let canon_oj = pair_map.sym.transform_9bit(board.obb[orig_j]);
        let canon_xk = pair_map.sym.transform_9bit(board.xbb[orig_k]);
        let canon_ok = pair_map.sym.transform_9bit(board.obb[orig_k]);

        let j_id = self.micro.to_id[((canon_xj as usize) << 9) | (canon_oj as usize)] as usize;
        let k_id = self.micro.to_id[((canon_xk as usize) << 9) | (canon_ok as usize)] as usize;

        if j_id >= TOTAL_MICRO_STATES || k_id >= TOTAL_MICRO_STATES {
            return None;
        }

        // 5. Query the memory-mapped slice
        let slice_filename = format!("tier7_orbit_{}_tau_{:03}.bin", pair_map.orbit_id, tau_idx);
        let mut map_guard = self.open_mmaps.lock().unwrap();
        if !map_guard.contains_key(&slice_filename) {
            let slice_file = File::open(self.dir.join(&slice_filename)).ok()?;
            let mmap = unsafe { Mmap::map(&slice_file).ok()? };
            if mmap.len() != SLICE_BYTE_SIZE {
                return None;
            }
            map_guard.insert(slice_filename.clone(), mmap);
        }
        let mmap = map_guard.get(&slice_filename)?;

        let byte = mmap[j_id * TOTAL_MICRO_STATES + k_id];

        // 6. Decode byte for target and turn
        // byte = v(X, j) + 3 * v(X, k) + 9 * v(O, j) + 27 * v(O, k)
        let v_x_j = EgtbOutcome::from(byte % 3);
        let v_x_k = EgtbOutcome::from((byte / 3) % 3);
        let v_o_j = EgtbOutcome::from((byte / 9) % 3);
        let v_o_k = EgtbOutcome::from((byte / 27) % 3);

        let is_target_j = if pair_map.flipped {
            board.last_super_index == k as u8
        } else {
            board.last_super_index == j as u8
        };
        let is_target_k = if pair_map.flipped {
            board.last_super_index == j as u8
        } else {
            board.last_super_index == k as u8
        };

        let outcome = if board.x_to_move {
            if is_target_j {
                v_x_j
            } else if is_target_k {
                v_x_k
            } else {
                // Free choice
                if v_x_j == EgtbOutcome::P1Win || v_x_k == EgtbOutcome::P1Win {
                    EgtbOutcome::P1Win
                } else if v_x_j == EgtbOutcome::Draw || v_x_k == EgtbOutcome::Draw {
                    EgtbOutcome::Draw
                } else {
                    EgtbOutcome::P2Win
                }
            }
        } else {
            if is_target_j {
                v_o_j
            } else if is_target_k {
                v_o_k
            } else {
                // Free choice
                if v_o_j == EgtbOutcome::P2Win || v_o_k == EgtbOutcome::P2Win {
                    EgtbOutcome::P2Win
                } else if v_o_j == EgtbOutcome::Draw || v_o_k == EgtbOutcome::Draw {
                    EgtbOutcome::Draw
                } else {
                    EgtbOutcome::P1Win
                }
            }
        };

        Some(outcome)
    }

    /// Direct query for retrograde induction from Tier 6 into Tier 7.
    /// Given two remaining open subboards (rem_a, rem_b) with rem_a < rem_b,
    /// their micro state IDs (micro_a_id, micro_b_id),
    /// the macro tau flags (xw_a, xw_b, ow_a, ow_b, x_shared, o_shared, delta),
    /// target (0 for rem_a, 1 for rem_b, 2 for Free Choice),
    /// and whether X is to move next (next_turn_is_x).
    pub fn probe_raw(
        &self,
        rem_a: usize,
        rem_b: usize,
        xw_a: bool,
        xw_b: bool,
        ow_a: bool,
        ow_b: bool,
        x_shared: bool,
        o_shared: bool,
        delta: i8,
        target: usize,
        next_turn_is_x: bool,
        micro_a_id: usize,
        micro_b_id: usize,
    ) -> Option<EgtbOutcome> {
        let pair_map = map_to_canonical_pair(rem_a, rem_b);

        // 1. Map tau to canonical orientation
        let (xw_j, xw_k, ow_j, ow_k) = if pair_map.flipped {
            (xw_b, xw_a, ow_b, ow_a)
        } else {
            (xw_a, xw_b, ow_a, ow_b)
        };

        let target_tau = Tier7MacroTau {
            xw_j,
            xw_k,
            ow_j,
            ow_k,
            x_shared,
            o_shared,
            delta: delta.clamp(-2, 2),
        };

        let taus = &self.orbit_taus[pair_map.orbit_id];
        let tau_idx = taus.binary_search(&target_tau).ok()?;

        // 2. Transform micro states under symmetry
        let (id_j, id_k) = if pair_map.flipped {
            (micro_b_id, micro_a_id)
        } else {
            (micro_a_id, micro_b_id)
        };

        let (xj, oj) = self.micro.states[id_j];
        let (xk, ok) = self.micro.states[id_k];

        let canon_xj = pair_map.sym.transform_9bit(xj);
        let canon_oj = pair_map.sym.transform_9bit(oj);
        let canon_xk = pair_map.sym.transform_9bit(xk);
        let canon_ok = pair_map.sym.transform_9bit(ok);

        let j_id = self.micro.to_id[((canon_xj as usize) << 9) | (canon_oj as usize)] as usize;
        let k_id = self.micro.to_id[((canon_xk as usize) << 9) | (canon_ok as usize)] as usize;

        if j_id >= TOTAL_MICRO_STATES || k_id >= TOTAL_MICRO_STATES {
            return None;
        }

        // 3. Query memory-mapped slice
        let slice_filename = format!("tier7_orbit_{}_tau_{:03}.bin", pair_map.orbit_id, tau_idx);
        let mut map_guard = self.open_mmaps.lock().unwrap();
        if !map_guard.contains_key(&slice_filename) {
            let slice_file = File::open(self.dir.join(&slice_filename)).ok()?;
            let mmap = unsafe { Mmap::map(&slice_file).ok()? };
            if mmap.len() != SLICE_BYTE_SIZE {
                return None;
            }
            map_guard.insert(slice_filename.clone(), mmap);
        }
        let mmap = map_guard.get(&slice_filename)?;

        let byte = mmap[j_id * TOTAL_MICRO_STATES + k_id];

        let v_x_j = EgtbOutcome::from(byte % 3);
        let v_x_k = EgtbOutcome::from((byte / 3) % 3);
        let v_o_j = EgtbOutcome::from((byte / 9) % 3);
        let v_o_k = EgtbOutcome::from((byte / 27) % 3);

        let is_target_j = if pair_map.flipped {
            target == 1
        } else {
            target == 0
        };
        let is_target_k = if pair_map.flipped {
            target == 0
        } else {
            target == 1
        };

        let outcome = if next_turn_is_x {
            if is_target_j {
                v_x_j
            } else if is_target_k {
                v_x_k
            } else {
                // Free choice
                if v_x_j == EgtbOutcome::P1Win || v_x_k == EgtbOutcome::P1Win {
                    EgtbOutcome::P1Win
                } else if v_x_j == EgtbOutcome::Draw || v_x_k == EgtbOutcome::Draw {
                    EgtbOutcome::Draw
                } else {
                    EgtbOutcome::P2Win
                }
            }
        } else {
            if is_target_j {
                v_o_j
            } else if is_target_k {
                v_o_k
            } else {
                // Free choice
                if v_o_j == EgtbOutcome::P2Win || v_o_k == EgtbOutcome::P2Win {
                    EgtbOutcome::P2Win
                } else if v_o_j == EgtbOutcome::Draw || v_o_k == EgtbOutcome::Draw {
                    EgtbOutcome::Draw
                } else {
                    EgtbOutcome::P1Win
                }
            }
        };

        Some(outcome)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_pair_reduction_all_36_pairs() {
        let mut orbit_counts = [0usize; 8];
        for j in 0..8 {
            for k in (j + 1)..9 {
                let map = map_to_canonical_pair(j, k);
                orbit_counts[map.orbit_id] += 1;
                let (rep_j, rep_k) = CANONICAL_ORBITS[map.orbit_id];
                assert_eq!(map.rep_j, rep_j);
                assert_eq!(map.rep_k, rep_k);

                // Verify symmetry mapping preserves pair
                let tj = map.sym.transform_3x3(j);
                let tk = map.sym.transform_3x3(k);
                if map.flipped {
                    assert_eq!((tk, tj), (rep_j, rep_k));
                } else {
                    assert_eq!((tj, tk), (rep_j, rep_k));
                }
            }
        }

        assert_eq!(orbit_counts, [8, 4, 4, 8, 2, 4, 4, 2]);
    }

    #[test]
    fn test_enumerate_orbit_taus_count() {
        let mut total = 0;
        for (orbit_id, &(rep_j, rep_k)) in CANONICAL_ORBITS.iter().enumerate() {
            let taus = enumerate_orbit_taus(rep_j, rep_k);
            println!("Orbit {}: {} taus", orbit_id, taus.len());
            total += taus.len();
        }
        assert_eq!(total, 614, "Total canonical macro vectors across all 8 orbits must be exactly 614");
    }

    #[test]
    fn test_single_canonical_slice_solve_accuracy() {
        let micro = MicroSubboardTable::build();
        let transitions = MicroTransitionTable::build(&micro);
        let tier8 = Tier8Lookup::new(&micro);
        let mut tables = allocate_pair_layer_tables(&micro);
        let mut slice = vec![0u8; SLICE_BYTE_SIZE];

        let tau = Tier7MacroTau {
            xw_j: false,
            xw_k: false,
            ow_j: false,
            ow_k: false,
            x_shared: false,
            o_shared: false,
            delta: 0,
        };

        let t0 = Instant::now();
        solve_single_canonical_slice(
            0,
            1,
            &tau,
            &micro,
            &transitions,
            &tier8,
            &mut tables,
            &mut slice,
        );
        let elapsed = t0.elapsed();
        println!("Computed canonical slice (0, 1) across 123.7M pairs in {:.2}s", elapsed.as_secs_f64());

        // Base case (E=0, both subboards full/tied): delta=0 => Draw (0)
        // byte = 0 + 3*0 + 9*0 + 27*0 = 0
        let id_0 = micro.states_by_empty[0][0];
        let byte_0 = slice[id_0 * TOTAL_MICRO_STATES + id_0];
        assert_eq!(byte_0, 0, "Empty delta=0 state must evaluate to Draw (0)");

        // Verify valid range: all values <= 80
        assert!(slice.iter().all(|&b| b <= 80), "All packed bytes must be in 0..=80");
    }

    #[test]
    fn test_tier7_zfs_prober_live_queries() {
        let dir = Path::new(TIER7_ZFS_DIR);
        if !dir.exists() || !dir.join("manifest.json").exists() {
            println!("Notice: Tier 7 ZFS directory not available, skipping live prober test.");
            return;
        }

        let prober = Tier7ZfsProber::new(dir).expect("Failed to create Tier 7 ZFS prober");
        let mut board = UltimateTicTacToeBoard::default();

        // Subboards 0, 1, 2, 3, 5, 6, 7 are full ties (CAT = 0x1FF)
        // Subboards 4 (center) and 8 (corner) remain open with 1 empty cell each
        for &s in &[0usize, 1, 2, 3, 5, 6, 7] {
            // Fill subboard with a non-winning tie:
            // X: squares 0, 1, 5, 6, 7 (mask = 1 + 2 + 32 + 64 + 128 = 227)
            // O: squares 2, 3, 4, 8 (mask = 4 + 8 + 16 + 256 = 284)
            // 227 | 284 = 511 = 0x1FF (full tie)
            board.xbb[s] = 227;
            board.obb[s] = 284;
        }
        board.x_supers = 0;
        board.o_supers = 0;

        // Subboard 4 has 1 square left (square 8)
        board.xbb[4] = 227 & !(1 << 7); // 4 stones
        board.obb[4] = 284;             // 4 stones
        // Subboard 8 has 1 square left (square 8)
        board.xbb[8] = 227 & !(1 << 7);
        board.obb[8] = 284;

        board.last_super_index = 4;
        board.x_to_move = true;

        let t0 = Instant::now();
        let outcome = prober.probe(&board);
        let elapsed = t0.elapsed();
        println!("Live Tier 7 ZFS probe result: {:?} in {:.2}ns", outcome, elapsed.as_secs_f64() * 1e9);
        assert!(outcome.is_some(), "Tier 7 position must be resolved by prober");

        // Verify probe_raw parity
        let id_4 = prober.micro.to_id[((board.xbb[4] as usize) << 9) | (board.obb[4] as usize)] as usize;
        let id_8 = prober.micro.to_id[((board.xbb[8] as usize) << 9) | (board.obb[8] as usize)] as usize;
        let t1 = Instant::now();
        let raw_outcome = prober.probe_raw(
            4, 8,
            false, false, false, false,
            false, false,
            0,
            0, // target is 4 (rem_a)
            true,
            id_4, id_8,
        );
        let elapsed_raw = t1.elapsed();
        println!("Live Tier 7 ZFS probe_raw result: {:?} in {:.2}ns", raw_outcome, elapsed_raw.as_secs_f64() * 1e9);
        assert_eq!(outcome, raw_outcome, "probe_raw must produce identical outcome to probe(&board)");
    }
}

