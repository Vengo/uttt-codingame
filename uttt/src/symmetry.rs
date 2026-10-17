use crate::game_solving::GameMove;
use crate::uttt_trait_solving::{parse_index, UltimateTicTacToeBoard, ZobristConstants};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DihedralSymmetry {
    Identity = 0,
    Rot90 = 1,
    Rot180 = 2,
    Rot270 = 3,
    FlipHorizontal = 4,
    FlipVertical = 5,
    ReflectMainDiag = 6,
    ReflectAntiDiag = 7,
}

pub const ALL_SYMMETRIES: [DihedralSymmetry; 8] = [
    DihedralSymmetry::Identity,
    DihedralSymmetry::Rot90,
    DihedralSymmetry::Rot180,
    DihedralSymmetry::Rot270,
    DihedralSymmetry::FlipHorizontal,
    DihedralSymmetry::FlipVertical,
    DihedralSymmetry::ReflectMainDiag,
    DihedralSymmetry::ReflectAntiDiag,
];

impl DihedralSymmetry {
    #[inline(always)]
    pub fn inverse(self) -> Self {
        match self {
            DihedralSymmetry::Identity => DihedralSymmetry::Identity,
            DihedralSymmetry::Rot90 => DihedralSymmetry::Rot270,
            DihedralSymmetry::Rot180 => DihedralSymmetry::Rot180,
            DihedralSymmetry::Rot270 => DihedralSymmetry::Rot90,
            DihedralSymmetry::FlipHorizontal => DihedralSymmetry::FlipHorizontal,
            DihedralSymmetry::FlipVertical => DihedralSymmetry::FlipVertical,
            DihedralSymmetry::ReflectMainDiag => DihedralSymmetry::ReflectMainDiag,
            DihedralSymmetry::ReflectAntiDiag => DihedralSymmetry::ReflectAntiDiag,
        }
    }

    /// Transforms a 3x3 index (0..9)
    #[inline(always)]
    pub fn transform_3x3(self, idx: usize) -> usize {
        let r = idx / 3;
        let c = idx % 3;
        let (nr, nc) = match self {
            DihedralSymmetry::Identity => (r, c),
            DihedralSymmetry::Rot90 => (c, 2 - r),
            DihedralSymmetry::Rot180 => (2 - r, 2 - c),
            DihedralSymmetry::Rot270 => (2 - c, r),
            DihedralSymmetry::FlipHorizontal => (r, 2 - c),
            DihedralSymmetry::FlipVertical => (2 - r, c),
            DihedralSymmetry::ReflectMainDiag => (c, r),
            DihedralSymmetry::ReflectAntiDiag => (2 - c, 2 - r),
        };
        nr * 3 + nc
    }

    /// Transforms a 9-bit bitboard
    #[inline(always)]
    pub fn transform_9bit(self, bb: u16) -> u16 {
        let mut out = 0u16;
        for i in 0..9 {
            if (bb & (1 << i)) != 0 {
                let ni = self.transform_3x3(i);
                out |= 1 << ni;
            }
        }
        out
    }

    /// Transforms a move index (0..81)
    #[inline(always)]
    pub fn transform_move_index(self, move_idx: u8) -> u8 {
        let parsed = parse_index(move_idx);
        let nb = self.transform_3x3(parsed.sub_board_index);
        let ns = self.transform_3x3(parsed.sub_square_index);
        (nb * 9 + ns) as u8
    }

    /// Transforms a GameMove
    #[inline(always)]
    pub fn transform_move(self, m: GameMove) -> GameMove {
        GameMove {
            x_to_move: m.x_to_move,
            move_index: self.transform_move_index(m.move_index),
        }
    }
}

pub struct SymmetryEngine;

impl SymmetryEngine {
    /// Transforms an UltimateTicTacToeBoard under a symmetry
    pub fn transform_board(
        board: &UltimateTicTacToeBoard,
        sym: DihedralSymmetry,
        zobrist_constants: &ZobristConstants,
    ) -> UltimateTicTacToeBoard {
        let mut new_xbb = [0u16; 9];
        let mut new_obb = [0u16; 9];

        for b in 0..9 {
            let nb = sym.transform_3x3(b);
            new_xbb[nb] = sym.transform_9bit(board.xbb[b]);
            new_obb[nb] = sym.transform_9bit(board.obb[b]);
        }

        let new_x_supers = sym.transform_9bit(board.x_supers);
        let new_o_supers = sym.transform_9bit(board.o_supers);

        let new_last_super = if board.last_super_index < 9 {
            sym.transform_3x3(board.last_super_index as usize) as u8
        } else {
            board.last_super_index
        };

        let mut new_board = UltimateTicTacToeBoard {
            xbb: new_xbb,
            obb: new_obb,
            x_supers: new_x_supers,
            o_supers: new_o_supers,
            last_super_index: new_last_super,
            x_to_move: board.x_to_move,
            hash_state: 0,
        };

        // Recompute hash state
        new_board.hash_state = new_board.compute_fresh_hash(zobrist_constants);
        new_board
    }

    /// Computes canonical form across all 8 symmetries.
    /// Returns `(canonical_board, symmetry_that_produced_it)`.
    pub fn canonical_form(
        board: &UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
    ) -> (UltimateTicTacToeBoard, DihedralSymmetry) {
        let mut best_board = board.clone();
        let mut best_sym = DihedralSymmetry::Identity;

        for &sym in &ALL_SYMMETRIES {
            let candidate = if sym == DihedralSymmetry::Identity {
                board.clone()
            } else {
                Self::transform_board(board, sym, zobrist_constants)
            };

            if Self::is_less(&candidate, &best_board) {
                best_board = candidate;
                best_sym = sym;
            }
        }

        (best_board, best_sym)
    }

    /// Canonical hash for fast table lookup
    #[inline]
    pub fn canonical_hash(
        board: &UltimateTicTacToeBoard,
        zobrist_constants: &ZobristConstants,
    ) -> (u64, DihedralSymmetry) {
        let (canonical_board, sym) = Self::canonical_form(board, zobrist_constants);
        (canonical_board.full_hash(), sym)
    }

    /// Lexicographical comparison for canonical determination
    fn is_less(a: &UltimateTicTacToeBoard, b: &UltimateTicTacToeBoard) -> bool {
        if a.last_super_index != b.last_super_index {
            return a.last_super_index < b.last_super_index;
        }
        if a.x_supers != b.x_supers {
            return a.x_supers < b.x_supers;
        }
        if a.o_supers != b.o_supers {
            return a.o_supers < b.o_supers;
        }
        if a.xbb != b.xbb {
            return a.xbb < b.xbb;
        }
        a.obb < b.obb
    }
}

impl UltimateTicTacToeBoard {
    /// Computes a full fresh hash from scratch including all pieces, side to move, and last_super_index
    pub fn compute_fresh_hash(&self, zobrist_constants: &ZobristConstants) -> u64 {
        let mut h = crate::uttt_trait_solving::HASH_DEFAULT;
        for b in 0..9 {
            for s in 0..9 {
                let mask = 1 << s;
                let cell_idx = b * 9 + s;
                if (self.xbb[b] & mask) != 0 {
                    h ^= zobrist_constants.x_constants[cell_idx];
                }
                if (self.obb[b] & mask) != 0 {
                    h ^= zobrist_constants.o_constants[cell_idx];
                }
            }
        }
        if self.x_to_move {
            h ^= zobrist_constants.x_to_move;
        }
        h
    }

    /// Returns the full unique hash combining piece hash with directed subboard
    #[inline(always)]
    pub fn full_hash(&self) -> u64 {
        self.hash_state ^ ((self.last_super_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetry_inverse() {
        for &sym in &ALL_SYMMETRIES {
            let inv = sym.inverse();
            for i in 0..9 {
                assert_eq!(inv.transform_3x3(sym.transform_3x3(i)), i);
            }
            for move_idx in 0..81 {
                assert_eq!(inv.transform_move_index(sym.transform_move_index(move_idx)), move_idx);
            }
        }
    }
}
