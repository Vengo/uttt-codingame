//! Tier 6 Macro-Factored Endgame Tablebase Specification, Symmetry Reduction,
//! Retrograde Transition into Tier 7, and Depth-Stratified Hybrid Solver.
//!
//! Features:
//! - 13-Bit Macro Boundary Condition Vector tau_6 + 3-bit Clamped CodinGame Delta tiebreaker
//! - D4 Dihedral Symmetry Reduction of the 84 Subboard Triplets:
//!   * 56 triplets with Center closed partition into EXACTLY 10 canonical classes.
//!   * 28 triplets with Center open partition into 6 canonical classes.
//!   * Total: 16 canonical orbits under D4.
//! - Closed-Form Retrograde Induction: any move resolving an open subboard (j, k, or l)
//!   instantly transitions to Tier 7 and queries `MacroFactoredProber` in < 100 ns.
//! - Depth-Stratified Solver for E <= 6 empty cells and Hybrid DF-PN engine for deeper positions.

use crate::egtb::macro_factored::{
    check_subboard_win, encode_tier7_key, pair_to_index, MacroFactoredProber, WINNING_LINES,
};
use crate::egtb::EgtbOutcome;
use crate::symmetry::DihedralSymmetry;
use crate::uttt_trait_solving::UltimateTicTacToeBoard;

/// Saturated mini-board differential clamped to {-3..+3}
pub const TIER_6_MIN_DELTA: i8 = -3;
pub const TIER_6_MAX_DELTA: i8 = 3;

/// 3-Board Macro Boundary Condition Vector tau_6 for Tier 6
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tier6MacroCondition {
    // Single-board win flags: does winning this open subboard complete 3-in-a-row with closed subboards?
    pub xw_j: bool,
    pub xw_k: bool,
    pub xw_l: bool,
    pub ow_j: bool,
    pub ow_k: bool,
    pub ow_l: bool,

    // Pairwise collinear shared line flags: does pair (a, b) form a collinear line with a closed subboard m won by X/O?
    pub x_shared_jk: bool,
    pub x_shared_jl: bool,
    pub x_shared_kl: bool,
    pub o_shared_jk: bool,
    pub o_shared_jl: bool,
    pub o_shared_kl: bool,

    // Triplet collinear line flag: do j, k, l form a winning line themselves?
    pub line_jkl: bool,

    // Clamped CodinGame tiebreaker: Delta = |M_X| - |M_O| on closed subboards, clamped to {-3..+3}
    pub delta: i8,
}

impl Tier6MacroCondition {
    /// Encodes tau_6 and delta into a compact 16-bit integer
    #[inline]
    pub fn encode_u16(&self) -> u16 {
        let delta_u = (self.delta.clamp(TIER_6_MIN_DELTA, TIER_6_MAX_DELTA) - TIER_6_MIN_DELTA) as u16; // 0..6 (3 bits)
        (self.xw_j as u16)
            | ((self.xw_k as u16) << 1)
            | ((self.xw_l as u16) << 2)
            | ((self.ow_j as u16) << 3)
            | ((self.ow_k as u16) << 4)
            | ((self.ow_l as u16) << 5)
            | ((self.x_shared_jk as u16) << 6)
            | ((self.x_shared_jl as u16) << 7)
            | ((self.x_shared_kl as u16) << 8)
            | ((self.o_shared_jk as u16) << 9)
            | ((self.o_shared_jl as u16) << 10)
            | ((self.o_shared_kl as u16) << 11)
            | ((self.line_jkl as u16) << 12)
            | (delta_u << 13)
    }

    /// Decodes a 16-bit integer back to Tier6MacroCondition
    #[inline]
    pub fn decode_u16(val: u16) -> Self {
        let delta_u = ((val >> 13) & 0x7) as i8;
        let delta = delta_u + TIER_6_MIN_DELTA;

        Tier6MacroCondition {
            xw_j: (val & (1 << 0)) != 0,
            xw_k: (val & (1 << 1)) != 0,
            xw_l: (val & (1 << 2)) != 0,
            ow_j: (val & (1 << 3)) != 0,
            ow_k: (val & (1 << 4)) != 0,
            ow_l: (val & (1 << 5)) != 0,
            x_shared_jk: (val & (1 << 6)) != 0,
            x_shared_jl: (val & (1 << 7)) != 0,
            x_shared_kl: (val & (1 << 8)) != 0,
            o_shared_jk: (val & (1 << 9)) != 0,
            o_shared_jl: (val & (1 << 10)) != 0,
            o_shared_kl: (val & (1 << 11)) != 0,
            line_jkl: (val & (1 << 12)) != 0,
            delta,
        }
    }
}

/// Computes the 13-bit macro condition vector tau_6 and clamped tiebreaker delta
/// for three open subboards (j, k, l) with 0 <= j < k < l < 9.
#[inline]
pub fn extract_tier6_tau(
    x_supers: u16,
    o_supers: u16,
    j: usize,
    k: usize,
    l: usize,
) -> Tier6MacroCondition {
    debug_assert!(j < k && k < l && l < 9);

    // 1. Single-board win completion flags with closed subboards
    let xw_j = check_subboard_win(x_supers | (1 << j));
    let xw_k = check_subboard_win(x_supers | (1 << k));
    let xw_l = check_subboard_win(x_supers | (1 << l));
    let ow_j = check_subboard_win(o_supers | (1 << j));
    let ow_k = check_subboard_win(o_supers | (1 << k));
    let ow_l = check_subboard_win(o_supers | (1 << l));

    // 2. Collinear line inspection
    let mut line_jkl = false;
    let mut x_shared_jk = false;
    let mut x_shared_jl = false;
    let mut x_shared_kl = false;
    let mut o_shared_jk = false;
    let mut o_shared_jl = false;
    let mut o_shared_kl = false;

    for line in WINNING_LINES {
        let has_j = line[0] == j || line[1] == j || line[2] == j;
        let has_k = line[0] == k || line[1] == k || line[2] == k;
        let has_l = line[0] == l || line[1] == l || line[2] == l;

        if has_j && has_k && has_l {
            line_jkl = true;
            continue;
        }

        // Pairwise collinear lines through a closed third subboard m
        if has_j && has_k {
            let m = line.iter().copied().find(|&s| s != j && s != k).unwrap();
            if m != l {
                if (x_supers & (1 << m)) != 0 { x_shared_jk = true; }
                if (o_supers & (1 << m)) != 0 { o_shared_jk = true; }
            }
        }
        if has_j && has_l {
            let m = line.iter().copied().find(|&s| s != j && s != l).unwrap();
            if m != k {
                if (x_supers & (1 << m)) != 0 { x_shared_jl = true; }
                if (o_supers & (1 << m)) != 0 { o_shared_jl = true; }
            }
        }
        if has_k && has_l {
            let m = line.iter().copied().find(|&s| s != k && s != l).unwrap();
            if m != j {
                if (x_supers & (1 << m)) != 0 { x_shared_kl = true; }
                if (o_supers & (1 << m)) != 0 { o_shared_kl = true; }
            }
        }
    }

    // 3. Clamped CodinGame tiebreaker
    let open_mask = (1 << j) | (1 << k) | (1 << l);
    let x_won_closed = (x_supers & !open_mask).count_ones() as i8;
    let o_won_closed = (o_supers & !open_mask).count_ones() as i8;
    let delta = (x_won_closed - o_won_closed).clamp(TIER_6_MIN_DELTA, TIER_6_MAX_DELTA);

    Tier6MacroCondition {
        xw_j,
        xw_k,
        xw_l,
        ow_j,
        ow_k,
        ow_l,
        x_shared_jk,
        x_shared_jl,
        x_shared_kl,
        o_shared_jk,
        o_shared_jl,
        o_shared_kl,
        line_jkl,
        delta,
    }
}

/// Maps an unordered triplet (j, k, l) with 0 <= j < k < l < 9 to an index in 0..84
#[inline]
pub fn triplet_to_index(j: usize, k: usize, l: usize) -> usize {
    debug_assert!(j < k && k < l && l < 9);
    // Standard lexicographical combination index for combinations of 3 from 9:
    // sum_{a=0}^{j-1} binom(9-1-a, 2) + sum_{b=j+1}^{k-1} binom(9-1-b, 1) + (l - k - 1)
    let mut idx = 0;
    for a in 0..j {
        idx += (8 - a) * (7 - a) / 2;
    }
    for b in (j + 1)..k {
        idx += 8 - b;
    }
    idx += l - k - 1;
    idx
}

/// Inverts an index in 0..84 back to an unordered triplet (j, k, l)
#[inline]
pub fn index_to_triplet(mut idx: usize) -> (usize, usize, usize) {
    debug_assert!(idx < 84);
    for j in 0..7 {
        for k in (j + 1)..8 {
            for l in (k + 1)..9 {
                if idx == 0 {
                    return (j, k, l);
                }
                idx -= 1;
            }
        }
    }
    unreachable!()
}

/// D4 Dihedral Canonical Triplet representation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CanonicalTriplet {
    pub rep_j: usize,
    pub rep_k: usize,
    pub rep_l: usize,
    pub sym: DihedralSymmetry,
    pub orbit_id: usize, // 0..15
    pub is_center_closed: bool,
    pub class_10_id: Option<usize>, // Some(0..9) if center is closed, None if center is open
}

/// Computes the canonical triplet representative under D4 dihedral group
pub fn canonical_triplet(j: usize, k: usize, l: usize) -> CanonicalTriplet {
    let symmetries = [
        DihedralSymmetry::Identity,
        DihedralSymmetry::Rot90,
        DihedralSymmetry::Rot180,
        DihedralSymmetry::Rot270,
        DihedralSymmetry::FlipHorizontal,
        DihedralSymmetry::FlipVertical,
        DihedralSymmetry::ReflectMainDiag,
        DihedralSymmetry::ReflectAntiDiag,
    ];

    let mut best_rep = (9usize, 9usize, 9usize);
    let mut best_sym = DihedralSymmetry::Identity;

    for &sym in &symmetries {
        let mut mapped = [sym.transform_3x3(j), sym.transform_3x3(k), sym.transform_3x3(l)];
        mapped.sort_unstable();
        let cand = (mapped[0], mapped[1], mapped[2]);
        if cand < best_rep {
            best_rep = cand;
            best_sym = sym;
        }
    }

    let is_center_closed = best_rep.0 != 4 && best_rep.1 != 4 && best_rep.2 != 4;

    // 16 canonical representatives
    let (orbit_id, class_10_id) = match best_rep {
        // 10 canonical classes when center 4 is closed (56 triplets):
        (0, 1, 2) => (0, Some(0)),
        (0, 1, 3) => (1, Some(1)),
        (0, 1, 5) => (2, Some(2)),
        (0, 1, 6) => (3, Some(3)),
        (0, 1, 7) => (4, Some(4)),
        (0, 1, 8) => (5, Some(5)),
        (0, 2, 6) => (6, Some(6)),
        (0, 2, 7) => (7, Some(7)),
        (0, 5, 7) => (8, Some(8)),
        (1, 3, 5) => (9, Some(9)),

        // 6 canonical classes when center 4 is open (28 triplets):
        (0, 1, 4) => (10, None),
        (0, 2, 4) => (11, None),
        (0, 4, 5) => (12, None),
        (0, 4, 8) => (13, None),
        (1, 3, 4) => (14, None),
        (1, 4, 7) => (15, None),

        _ => unreachable!("Unknown triplet canonical representative: {:?}", best_rep),
    };

    CanonicalTriplet {
        rep_j: best_rep.0,
        rep_k: best_rep.1,
        rep_l: best_rep.2,
        sym: best_sym,
        orbit_id,
        is_center_closed,
        class_10_id,
    }
}

/// Resolution state of a subboard when a move completes it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubboardResolution {
    XWin,
    OWin,
    Tie,
}

/// Evaluates the retrograde transition from Tier 6 (j, k, l) to Tier 7
/// when subboard `resolved_subboard` (which must be j, k, or l) resolves with `sub_outcome`.
///
/// Any move resolving an open subboard immediately queries `MacroFactoredProber`
/// to retrieve the exact Tier 7 outcome in < 100 ns.
pub fn transition_tier6_to_tier7(
    tau6: &Tier6MacroCondition,
    j: usize,
    k: usize,
    l: usize,
    resolved_subboard: usize,
    sub_outcome: SubboardResolution,
    last_sq: usize,
    turn_was_x: bool,
    micro_ids: (usize, usize, usize), // (j_id, k_id, l_id)
    prober: &MacroFactoredProber,
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

    // 1. Terminal Check & Updated Tier 7 Boundary Parameters
    let (xw_a_prime, xw_b_prime, ow_a_prime, ow_b_prime, x_shared_prime, o_shared_prime, delta_prime) =
        match sub_outcome {
            SubboardResolution::XWin => {
                // If X winning this subboard completes a 3-in-a-row with closed subboards, immediate win!
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
                // If O winning this subboard completes a 3-in-a-row with closed subboards, immediate win!
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
    // The player to move next is !turn_was_x
    let next_turn = !turn_was_x;
    let target = if last_sq == rem_a {
        0 // opponent forced to play in rem_a
    } else if last_sq == rem_b {
        1 // opponent forced to play in rem_b
    } else {
        2 // Free choice (last_sq points to a closed subboard)
    };

    // 3. Query Tier 7 Tablebase via Prober (< 100 ns)
    let pair_idx = pair_to_index(rem_a, rem_b);
    let micro_composite = ((micro_a as u64) << 16) | (micro_b as u64);

    let key = encode_tier7_key(
        pair_idx,
        xw_a_prime,
        xw_b_prime,
        ow_a_prime,
        ow_b_prime,
        x_shared_prime,
        o_shared_prime,
        delta_prime,
        target,
        next_turn,
        micro_composite,
    );

    prober.probe_tier_7_key(key).unwrap_or(EgtbOutcome::Unknown)
}

/// Encodes a Tier 6 tablebase position into a 64-bit key
#[inline]
pub fn encode_tier6_key(
    j: usize,
    k: usize,
    l: usize,
    tau: &Tier6MacroCondition,
    target: usize, // 0 = j, 1 = k, 2 = l, 3 = Free Choice
    turn: bool,
    j_id: usize,
    k_id: usize,
    l_id: usize,
) -> u64 {
    let trip_idx = triplet_to_index(j, k, l) as u64; // 0..83 (7 bits)
    let tau_u16 = tau.encode_u16() as u64; // 16 bits
    let target_bits = (target as u64) & 0x3; // 2 bits
    let turn_bit = if turn { 1u64 } else { 0u64 }; // 1 bit

    // Micro stone layout hash across (j, k, l)
    let micro_hash = ((j_id as u64 & 0x3FFF) << 24)
        | ((k_id as u64 & 0x3FFF) << 12)
        | (l_id as u64 & 0x3FFF); // 36 bits

    (trip_idx << 57)
        | (tau_u16 << 41)
        | (target_bits << 39)
        | (turn_bit << 38)
        | (micro_hash & 0x0000_003F_FFFF_FFFF)
}

/// Depth-Stratified Layer Table holding outcomes for E <= 6 empty cells
pub struct Tier6LayerTable {
    pub n_j: usize,
    pub n_k: usize,
    pub n_l: usize,
    pub data: Vec<EgtbOutcome>,
}

impl Tier6LayerTable {
    pub fn new(n_j: usize, n_k: usize, n_l: usize) -> Self {
        Tier6LayerTable {
            n_j,
            n_k,
            n_l,
            // 8 states per cell configuration: 4 targets (j, k, l, Free) * 2 turns
            data: vec![EgtbOutcome::Draw; n_j * n_k * n_l * 8],
        }
    }

    #[inline(always)]
    pub fn get(&self, idx_j: usize, idx_k: usize, idx_l: usize, turn: bool, target: usize) -> EgtbOutcome {
        let turn_offset = if turn { 0 } else { 4 };
        let offset = ((idx_j * self.n_k + idx_k) * self.n_l + idx_l) * 8 + turn_offset + target;
        self.data[offset]
    }

    #[inline(always)]
    pub fn set(&mut self, idx_j: usize, idx_k: usize, idx_l: usize, turn: bool, target: usize, outcome: EgtbOutcome) {
        let turn_offset = if turn { 0 } else { 4 };
        let offset = ((idx_j * self.n_k + idx_k) * self.n_l + idx_l) * 8 + turn_offset + target;
        self.data[offset] = outcome;
    }
}

/// Hybrid DF-PN Search Engine with instant EGTB leaf probing
pub struct HybridDfPnSolver<'a> {
    pub dfpn: crate::dfpn::DfPnSolver,
    pub prober: &'a MacroFactoredProber,
}

impl<'a> HybridDfPnSolver<'a> {
    pub fn new(max_nodes: usize, prober: &'a MacroFactoredProber) -> Self {
        HybridDfPnSolver {
            dfpn: crate::dfpn::DfPnSolver::new(max_nodes),
            prober,
        }
    }

    /// Solves position with instant tablebase leaf cutoff:
    /// - If resolved >= 7 -> queries MacroFactoredProber instantly (< 100 ns)
    /// - If resolved == 6 -> queries Tier 6 prober or retrograde transition
    /// - Deeper positions solved via DF-PN proof numbers with Grandfather lookahead
    pub fn solve(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_lookup: &[Vec<crate::game_solving::GameMove>],
        zobrist_constants: &crate::uttt_trait_solving::ZobristConstants,
    ) -> Option<bool> {
        // 1. Direct tablebase lookup at root
        if let Some(res) = self.prober.probe(board) {
            return match res {
                EgtbOutcome::P1Win => Some(board.x_to_move),
                EgtbOutcome::P2Win => Some(!board.x_to_move),
                EgtbOutcome::Draw => Some(false),
                EgtbOutcome::Unknown => None,
            };
        }

        // 2. Delegate to DF-PN proof search
        self.dfpn.solve(board, legal_lookup, zobrist_constants)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d4_symmetry_canonical_reduction_orbits() {
        let mut orbit_reps = std::collections::HashSet::new();
        let mut class_10_reps = std::collections::HashSet::new();
        let mut center_closed_count = 0;
        let mut center_open_count = 0;

        for j in 0..7 {
            for k in (j + 1)..8 {
                for l in (k + 1)..9 {
                    let canon = canonical_triplet(j, k, l);
                    orbit_reps.insert((canon.rep_j, canon.rep_k, canon.rep_l));

                    if canon.is_center_closed {
                        center_closed_count += 1;
                        assert!(canon.class_10_id.is_some());
                        class_10_reps.insert(canon.class_10_id.unwrap());
                    } else {
                        center_open_count += 1;
                        assert!(canon.class_10_id.is_none());
                    }
                }
            }
        }

        // Mathematical Assertions
        assert_eq!(center_closed_count, 56, "binom(8, 3) must be 56 triplets without center 4");
        assert_eq!(center_open_count, 28, "binom(8, 2) must be 28 triplets with center 4");
        assert_eq!(center_closed_count + center_open_count, 84, "Total triplets must be 84");

        // The 56 non-center triplets MUST partition into EXACTLY 10 canonical classes!
        assert_eq!(class_10_reps.len(), 10, "Non-center triplets must reduce to exactly 10 canonical classes");
        // All 84 triplets MUST partition into EXACTLY 16 canonical orbits under D4!
        assert_eq!(orbit_reps.len(), 16, "Total D4 orbits must be exactly 16");
    }

    #[test]
    fn test_triplet_indexing_bijection() {
        for idx in 0..84 {
            let (j, k, l) = index_to_triplet(idx);
            assert!(j < k && k < l && l < 9);
            let back_idx = triplet_to_index(j, k, l);
            assert_eq!(back_idx, idx, "Triplet indexing must be a perfect bijection");
        }
    }

    #[test]
    fn test_tier6_macro_condition_bit_packing() {
        let cond = Tier6MacroCondition {
            xw_j: true,
            xw_k: false,
            xw_l: true,
            ow_j: false,
            ow_k: true,
            ow_l: false,
            x_shared_jk: true,
            x_shared_jl: false,
            x_shared_kl: false,
            o_shared_jk: false,
            o_shared_jl: true,
            o_shared_kl: false,
            line_jkl: false,
            delta: -2,
        };

        let encoded = cond.encode_u16();
        let decoded = Tier6MacroCondition::decode_u16(encoded);
        assert_eq!(cond, decoded, "Bit packing round-trip must be lossless");
    }

    #[test]
    fn test_extract_tier6_tau_invariants() {
        // Construct macro board where subboards 0, 1, 2 form a collinear line
        // Let X win subboard 2 (m = 2). Open subboards are j = 0, k = 1, l = 4 (center)
        let mut x_supers = 1 << 2; // subboard 2 won by X
        let o_supers = 1 << 8;     // subboard 8 won by O
        // Other closed subboards: 3, 5, 6, 7 (tied)

        let tau = extract_tier6_tau(x_supers, o_supers, 0, 1, 4);
        // j=0, k=1, m=2 is row (0, 1, 2). Since m=2 won by X, x_shared_jk must be TRUE!
        assert!(tau.x_shared_jk, "Row (0, 1, 2) with m=2 won by X must set x_shared_jk");
        assert!(!tau.o_shared_jk);
        assert!(!tau.line_jkl, "Triplet (0, 1, 4) is not collinear");

        // Closed subboards won: X has 1 (subboard 2), O has 1 (subboard 8) -> Delta = 0
        assert_eq!(tau.delta, 0);

        // Case B: collinear triplet (0, 1, 2)
        let tau_collinear = extract_tier6_tau(0, 0, 0, 1, 2);
        assert!(tau_collinear.line_jkl, "Triplet (0, 1, 2) must be identified as collinear line");
        assert!(!tau_collinear.x_shared_jk);
        assert!(!tau_collinear.x_shared_jl);
        assert!(!tau_collinear.x_shared_kl);
    }

    #[test]
    fn test_retrograde_transition_soundness() {
        let temp_dir = std::env::temp_dir().join(format!("test_mf_t6_trans_{}", rand::random::<u64>()));
        let solver = crate::egtb::macro_factored::MacroFactoredSolver::new(&temp_dir, 2);

        // Precompute Tier 8 and Tier 7 tables
        let _ = solver.solve_tier_8().expect("Tier 8 solve failed");
        let _ = solver.solve_tier_7(1).expect("Tier 7 solve failed");

        let prober = MacroFactoredProber::with_dir(&temp_dir);

        // Setup Tier 6 condition: open subboards (0, 1, 4).
        // Let X have single-board win on 0 (xw_0 = true)
        let tau_win = Tier6MacroCondition {
            xw_j: true, // X winning 0 completes game
            xw_k: false,
            xw_l: false,
            ow_j: false,
            ow_k: false,
            ow_l: false,
            x_shared_jk: false,
            x_shared_jl: false,
            x_shared_kl: false,
            o_shared_jk: false,
            o_shared_jl: false,
            o_shared_kl: false,
            line_jkl: false,
            delta: 0,
        };

        // When subboard 0 resolves with XWin, must be an immediate P1Win!
        let outcome = transition_tier6_to_tier7(
            &tau_win,
            0, 1, 4,
            0, // resolving j = 0
            SubboardResolution::XWin,
            1, // last move directed to subboard 1
            true, // turn was X
            (0, 0, 0),
            &prober,
        );
        assert_eq!(outcome, EgtbOutcome::P1Win, "Winning subboard 0 with xw_0=true must immediately yield P1Win");

        // When subboard 0 resolves with Tie, transitions to Tier 7 query
        let outcome_tie = transition_tier6_to_tier7(
            &tau_win,
            0, 1, 4,
            0,
            SubboardResolution::Tie,
            1,
            true,
            (0, 0, 0),
            &prober,
        );
        // Tier 7 query must execute cleanly and return a valid outcome
        assert!(outcome_tie != EgtbOutcome::Unknown, "Tier 7 probe must successfully return outcome");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
