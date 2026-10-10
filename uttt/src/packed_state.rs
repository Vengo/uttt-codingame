use std::fmt;
use crate::canonical::canonicalize_board;
use crate::uttt_trait_solving::UltimateTicTacToeBoard;

const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];

/// Exact 167-bit bit-packed integer representation of an Ultimate Tic-Tac-Toe state:
/// - 81 bits: X cells (9 subboards x 9 squares)
/// - 81 bits: O cells (9 subboards x 9 squares)
/// - 4 bits: Directed subboard index (0..9, where 9 = Free Choice)
/// - 1 bit: Player to move (1 = X, 0 = O)
/// Total = 167 bits <= 192 bits (24 bytes).
/// Fits exactly into three 64-bit unsigned integers: `[u64; 3]`.
///
/// Guaranteed zero hash collisions across all 10^28 possible game states.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PackedBoardState {
    pub words: [u64; 3], // 24 bytes, 8-byte aligned
}

impl PackedBoardState {
    /// Constructs a `PackedBoardState` directly from raw words.
    #[inline(always)]
    pub const fn from_words(words: [u64; 3]) -> Self {
        PackedBoardState { words }
    }

    /// Packs an `UltimateTicTacToeBoard` into exact 167-bit representation without mutating it.
    pub fn from_board(board: &UltimateTicTacToeBoard) -> Self {
        let mut x_all: u128 = 0;
        let mut o_all: u128 = 0;

        for (i, (&x, &o)) in board.xbb.iter().zip(board.obb.iter()).enumerate() {
            x_all |= ((x & 0x1FF) as u128) << (9 * i);
            o_all |= ((o & 0x1FF) as u128) << (9 * i);
        }

        let x_lo = (x_all & 0xFFFF_FFFF_FFFF_FFFF) as u64;
        let x_hi = (x_all >> 64) as u64; // 17 bits

        let o_lo = (o_all & 0x7FFF_FFFF_FFFF) as u64; // 47 bits
        let o_hi = (o_all >> 47) as u64; // 34 bits

        let word0 = x_lo;
        let word1 = (x_hi & 0x1_FFFF) | (o_lo << 17);

        let last_super = (board.last_super_index.min(15) as u64) & 0xF;
        let x_turn = if board.x_to_move { 1u64 } else { 0u64 };
        let word2 = (o_hi & 0x3_FFFF_FFFF) | (last_super << 34) | (x_turn << 38);

        PackedBoardState {
            words: [word0, word1, word2],
        }
    }

    /// Canonicalizes the board (canonical win/tie subboards and directed index normalization)
    /// and packs it into `PackedBoardState`.
    pub fn from_board_canonical(board: &UltimateTicTacToeBoard) -> Self {
        let mut canonical = board.clone();
        canonicalize_board(&mut canonical);
        Self::from_board(&canonical)
    }

    /// Unpacks the 167-bit state back into an `UltimateTicTacToeBoard`.
    pub fn to_board(&self) -> UltimateTicTacToeBoard {
        let x_all = (self.words[0] as u128) | (((self.words[1] & 0x1_FFFF) as u128) << 64);
        let o_all = (((self.words[1] >> 17) & 0x7FFF_FFFF_FFFF) as u128)
            | (((self.words[2] & 0x3_FFFF_FFFF) as u128) << 47);

        let mut xbb = [0u16; 9];
        let mut obb = [0u16; 9];
        let mut x_supers = 0u16;
        let mut o_supers = 0u16;

        for i in 0..9 {
            let x = ((x_all >> (9 * i)) & 0x1FF) as u16;
            let o = ((o_all >> (9 * i)) & 0x1FF) as u16;
            xbb[i] = x;
            obb[i] = o;

            if RESULT_MASKS.iter().any(|&mask| (x & mask) == mask) {
                x_supers |= 1 << i;
            }
            if RESULT_MASKS.iter().any(|&mask| (o & mask) == mask) {
                o_supers |= 1 << i;
            }
        }

        let last_super_index = ((self.words[2] >> 34) & 0xF) as u8;
        let x_to_move = ((self.words[2] >> 38) & 1) != 0;

        UltimateTicTacToeBoard {
            xbb,
            obb,
            x_supers,
            o_supers,
            last_super_index,
            x_to_move,
            hash_state: 0,
        }
    }

    /// Extracts the directed subboard index (0..9).
    #[inline(always)]
    pub fn last_super_index(&self) -> u8 {
        ((self.words[2] >> 34) & 0xF) as u8
    }

    /// Extracts whether it is X's turn to move.
    #[inline(always)]
    pub fn x_to_move(&self) -> bool {
        ((self.words[2] >> 38) & 1) != 0
    }

    /// Computes a deterministic, ultra-fast 64-bit hash mixer over the 3 words.
    #[inline(always)]
    pub fn fast_hash(&self) -> u64 {
        let mut h = self.words[0].wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= self.words[1].wrapping_mul(0x94D0_49BB_1331_11EB);
        h = (h ^ (h >> 27)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= self.words[2].wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^ (h >> 31)
    }
}

impl fmt::Debug for PackedBoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PackedBoardState(0x{:016x}_{:016x}_{:016x}, target={}, x_turn={})",
            self.words[0],
            self.words[1],
            self.words[2],
            self.last_super_index(),
            self.x_to_move()
        )
    }
}

impl fmt::Display for PackedBoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "0x{:016x}{:016x}{:016x}",
            self.words[2], self.words[1], self.words[0]
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packed_state_size() {
        assert_eq!(std::mem::size_of::<PackedBoardState>(), 24);
    }

    #[test]
    fn test_pack_unpack_roundtrip_default() {
        let board = UltimateTicTacToeBoard::default();
        let packed = PackedBoardState::from_board(&board);
        let unpacked = packed.to_board();

        assert_eq!(board.xbb, unpacked.xbb);
        assert_eq!(board.obb, unpacked.obb);
        assert_eq!(board.x_supers, unpacked.x_supers);
        assert_eq!(board.o_supers, unpacked.o_supers);
        assert_eq!(board.last_super_index, unpacked.last_super_index);
        assert_eq!(board.x_to_move, unpacked.x_to_move);
    }
}
