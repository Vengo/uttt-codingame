use crate::uttt_trait_solving::UltimateTicTacToeBoard;

/// Canonical bitboards for resolved subboards:
/// - X win: all 9 cells marked as X (0x1FF)
/// - O win: all 9 cells marked as O (0x1FF)
/// - CAT (Tie): a valid full 3x3 pattern with no 3-in-a-row for either player.
pub const CANONICAL_X_WIN: u16 = 0x1FF;
pub const CANONICAL_O_WIN: u16 = 0x1FF;
pub const CANONICAL_TIE_X: u16 = 0x0E3; // 0b011100011 (5 cells)
pub const CANONICAL_TIE_O: u16 = 0x11C; // 0b100011100 (4 cells, disjoint from X, no 3-in-a-row)

pub const FREE_CHOICE_INDEX: u8 = 9;


/// Normalizes the directed target subboard (`last_super_index`).
/// In Ultimate Tic-Tac-Toe, if a player is directed to a subboard that has already
/// been resolved (won by X, won by O, or tied), that player receives free choice
/// across all remaining active subboards.
/// Normalizing `last_super_index` to 9 merges all such equivalent branches into a single state.
#[inline]
pub fn normalize_directed_subboard(board: &mut UltimateTicTacToeBoard) {
    if board.last_super_index < 9 {
        let resolved = board.x_supers | board.o_supers | board.get_tied_mask();
        if (resolved & (1 << board.last_super_index)) != 0 {
            board.last_super_index = FREE_CHOICE_INDEX;
        }
    }
}

/// Applies canonical transformations in-place to any subboards that have resolved.
/// - Any subboard won by X becomes `xbb = CANONICAL_X_WIN, obb = 0`.
/// - Any subboard won by O becomes `obb = CANONICAL_O_WIN, xbb = 0`.
/// - Any subboard tied becomes `xbb = CANONICAL_TIE_X, obb = CANONICAL_TIE_O`.
/// - If `last_super_index` points to a resolved subboard, it is normalized to 9.
pub fn canonicalize_board(board: &mut UltimateTicTacToeBoard) {
    let tied_mask = board.get_tied_mask();

    for i in 0..9 {
        let bit = 1u16 << i;
        if (board.x_supers & bit) != 0 {
            board.xbb[i] = CANONICAL_X_WIN;
            board.obb[i] = 0;
        } else if (board.o_supers & bit) != 0 {
            board.xbb[i] = 0;
            board.obb[i] = CANONICAL_O_WIN;
        } else if (tied_mask & bit) != 0 {
            board.xbb[i] = CANONICAL_TIE_X;
            board.obb[i] = CANONICAL_TIE_O;
        }
    }

    normalize_directed_subboard(board);
}

/// Non-destructive version: returns a new canonical `UltimateTicTacToeBoard`.
pub fn to_canonical_board(board: &UltimateTicTacToeBoard) -> UltimateTicTacToeBoard {
    let mut cloned = board.clone();
    canonicalize_board(&mut cloned);
    cloned
}

#[cfg(test)]
mod tests {
    use super::*;
    const CAT: u16 = (1 << 9) - 1;

    #[test]

    fn test_canonical_tie_pattern_validity() {
        assert_eq!(CANONICAL_TIE_X | CANONICAL_TIE_O, CAT);
        assert_eq!(CANONICAL_TIE_X & CANONICAL_TIE_O, 0);

        // Verify neither player has a winning line
        const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
        for mask in RESULT_MASKS {
            assert_ne!(CANONICAL_TIE_X & mask, mask, "X should not have won in canonical tie pattern");
            assert_ne!(CANONICAL_TIE_O & mask, mask, "O should not have won in canonical tie pattern");
        }
    }
}
