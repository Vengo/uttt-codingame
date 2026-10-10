use rand::{rngs::StdRng, Rng, SeedableRng};
use uttt::canonical::{
    CANONICAL_O_WIN, CANONICAL_TIE_O, CANONICAL_TIE_X, CANONICAL_X_WIN, FREE_CHOICE_INDEX,
};

use uttt::dfpn::DfPnSolver;
use uttt::game_solving::{AlphaBetaCompatible, GameResult};
use uttt::packed_state::PackedBoardState;
use uttt::uttt_trait_solving::UltimateTicTacToeBoard;

#[test]
fn test_packed_state_size_and_alignment() {
    assert_eq!(std::mem::size_of::<PackedBoardState>(), 24);
    assert_eq!(std::mem::align_of::<PackedBoardState>(), 8);
}

#[test]
fn test_tied_mask_guard_with_canonical_won_subboards() {
    let mut board = UltimateTicTacToeBoard::default();
    // Mark subboard 0 as won by X with canonical all 9 Xs
    board.xbb[0] = CANONICAL_X_WIN;
    board.obb[0] = 0;
    board.x_supers = 1; // subboard 0 won by X

    // get_tied_mask must NOT classify subboard 0 as tied
    let tied = board.get_tied_mask();
    assert_eq!(tied & 1, 0, "Won subboard 0 must not be classified as tied");

    // Now test O win
    board.xbb[1] = 0;
    board.obb[1] = CANONICAL_O_WIN;
    board.o_supers = 2; // subboard 1 won by O
    let tied = board.get_tied_mask();
    assert_eq!(tied & 2, 0, "Won subboard 1 must not be classified as tied");

    // Now test legitimate tie
    board.xbb[2] = CANONICAL_TIE_X;
    board.obb[2] = CANONICAL_TIE_O;
    let tied = board.get_tied_mask();
    assert_ne!(tied & 4, 0, "Canonical tie pattern must be recognized as tied");
}

#[test]
fn test_pack_unpack_roundtrip_random_playouts() {
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let mut rng = StdRng::seed_from_u64(0xDEAD_BEEF_CAFE);

    for game_idx in 0..100 {
        let mut board = UltimateTicTacToeBoard::default();

        while board.get_win_state() == GameResult::Unresolved {
            // Verify roundtrip fidelity on current board
            let packed = PackedBoardState::from_board(&board);
            let unpacked = packed.to_board();

            assert_eq!(board.xbb, unpacked.xbb, "Game {}: xbb mismatch", game_idx);
            assert_eq!(board.obb, unpacked.obb, "Game {}: obb mismatch", game_idx);
            assert_eq!(board.x_supers, unpacked.x_supers, "Game {}: x_supers mismatch", game_idx);
            assert_eq!(board.o_supers, unpacked.o_supers, "Game {}: o_supers mismatch", game_idx);
            assert_eq!(board.last_super_index, unpacked.last_super_index, "Game {}: last_super mismatch", game_idx);
            assert_eq!(board.x_to_move, unpacked.x_to_move, "Game {}: turn mismatch", game_idx);

            // Also verify fast_hash determinism
            assert_eq!(packed.fast_hash(), PackedBoardState::from_board(&unpacked).fast_hash());

            let moves = board.get_legal_moves(&legal_lookup);
            if moves.is_empty() {
                break;
            }
            let chosen = moves[rng.gen_range(0..moves.len())];
            board.make_move(chosen, &zobrist_constants);
        }
    }
}

#[test]
fn test_canonical_invariance_across_random_playouts() {
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let mut rng = StdRng::seed_from_u64(0x1234_5678_9ABC);

    for game_idx in 0..50 {
        let mut board = UltimateTicTacToeBoard::default();

        while board.get_win_state() == GameResult::Unresolved {
            let canon = board.to_canonical();

            // Invariant 1: Win state equivalence
            assert_eq!(
                board.get_win_state(),
                canon.get_win_state(),
                "Game {}: win state mismatch",
                game_idx
            );

            // Invariant 2: Score equivalence
            assert_eq!(
                board.score(),
                canon.score(),
                "Game {}: score mismatch",
                game_idx
            );

            // Invariant 3: Legal moves equivalence
            let raw_moves = board.get_legal_moves(&legal_lookup);
            let canon_moves = canon.get_legal_moves(&legal_lookup);
            assert_eq!(
                raw_moves, canon_moves,
                "Game {}: legal moves mismatch between raw and canonical",
                game_idx
            );

            // Invariant 4: Packed canonical is idempotent
            let packed_canon_1 = board.to_packed_canonical();
            let packed_canon_2 = canon.to_packed_canonical();
            assert_eq!(packed_canon_1, packed_canon_2);

            let moves = raw_moves;
            if moves.is_empty() {
                break;
            }
            let chosen = moves[rng.gen_range(0..moves.len())];
            board.make_move(chosen, &zobrist_constants);
        }
    }
}

#[test]
fn test_transposition_collapse_different_win_lines() {
    // Board A: Subboard 0 won by X via top row: squares 0, 1, 2 (xbb = 7)
    let mut board_a = UltimateTicTacToeBoard::default();
    board_a.xbb[0] = 7; // row 0: squares 0, 1, 2
    board_a.obb[0] = 56; // row 1: squares 3, 4, 5
    board_a.x_supers = 1;
    board_a.last_super_index = 4;
    board_a.x_to_move = false;

    // Board B: Subboard 0 won by X via diagonal: squares 0, 4, 8 (xbb = 273)
    let mut board_b = UltimateTicTacToeBoard::default();
    board_b.xbb[0] = 273; // diag: squares 0, 4, 8
    board_b.obb[0] = 56;
    board_b.x_supers = 1;
    board_b.last_super_index = 4;
    board_b.x_to_move = false;

    // Raw boards are distinctly different
    assert_ne!(board_a.xbb[0], board_b.xbb[0]);
    assert_ne!(board_a.to_packed(), board_b.to_packed());

    // In canonical form, both collapse to the exact same PackedBoardState!
    let packed_a = board_a.to_packed_canonical();
    let packed_b = board_b.to_packed_canonical();

    assert_eq!(packed_a, packed_b, "Different subboard win lines must collapse to the identical canonical packed state");
    assert_eq!(packed_a.fast_hash(), packed_b.fast_hash());

    // Check that subboard 0 in canonical form has all 9 Xs and 0 Os
    let unpacked = packed_a.to_board();
    assert_eq!(unpacked.xbb[0], CANONICAL_X_WIN);
    assert_eq!(unpacked.obb[0], 0);
}

#[test]
fn test_directed_subboard_normalization_collapse() {
    let mut board = UltimateTicTacToeBoard::default();
    // Subboards 0 and 1 are resolved (won by X)
    board.xbb[0] = CANONICAL_X_WIN;
    board.xbb[1] = CANONICAL_X_WIN;
    board.x_supers = 0b11; // subboards 0 and 1 won

    // State 1: Opponent is directed to subboard 0 (which is resolved)
    let mut state1 = board.clone();
    state1.last_super_index = 0;

    // State 2: Opponent is directed to subboard 1 (which is resolved)
    let mut state2 = board.clone();
    state2.last_super_index = 1;

    // State 3: Opponent receives free choice (index 9)
    let mut state3 = board.clone();
    state3.last_super_index = FREE_CHOICE_INDEX;

    // Raw states are different
    assert_ne!(state1.to_packed(), state2.to_packed());
    assert_ne!(state1.to_packed(), state3.to_packed());

    // Under canonicalization, all three states normalize to target = 9 and collapse!
    let p1 = state1.to_packed_canonical();
    let p2 = state2.to_packed_canonical();
    let p3 = state3.to_packed_canonical();

    assert_eq!(p1, p2);
    assert_eq!(p2, p3);
    assert_eq!(p1.last_super_index(), FREE_CHOICE_INDEX);
}

#[test]
fn test_dfpn_solver_with_packed_state_and_grandfather() {
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut solver = DfPnSolver::new(10_000);

    // 1. Terminal win state: P1 already won
    let mut board = UltimateTicTacToeBoard::default();
    board.x_supers = 7; // subboards 0, 1, 2 won by X (horizontal line)
    board.x_to_move = true;

    let res = solver.solve(&board, &legal_lookup, &zobrist_constants);
    assert_eq!(res, Some(true), "Immediate win should be proven");

    // 2. 1-ply winning move available
    let mut board = UltimateTicTacToeBoard::default();
    board.x_supers = 3; // subboards 0, 1 won by X. Subboard 2 is open.
    board.last_super_index = 2; // Directed to subboard 2
    board.xbb[2] = 3; // In subboard 2, X has squares 0 and 1. Square 2 wins subboard 2!
    board.x_to_move = true;

    solver.reset();
    let res = solver.solve(&board, &legal_lookup, &zobrist_constants);
    assert_eq!(res, Some(true), "1-ply win for X should be proven");
    assert!(solver.nodes_visited <= 5, "Grandfather/1-ply lookahead should solve in very few nodes");
}

#[test]
fn test_opening_book_harvesting_and_growth() {
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let temp_book_path = std::env::temp_dir().join(format!("test_book_{}.bin", rand::random::<u64>()));
    let mut book = uttt::opening_book::OpeningBook::with_path(&temp_book_path);

    // Initial state: empty book
    assert_eq!(book.entries.len(), 0);

    // Deepen cycle 1: Root search & tree harvesting
    let added_1 = book.deepen_book(
        2,
        std::time::Duration::from_millis(60),
        10, // low visit threshold for quick unit test
        &legal_lookup,
        &zobrist_constants,
    );
    assert!(added_1 > 0, "Cycle 1 should harvest positions from root search");
    let count_1 = book.entries.len();
    assert!(count_1 >= added_1);

    // Lookup root board should now succeed
    let root = UltimateTicTacToeBoard::default();
    let root_move = book.lookup(&root, &zobrist_constants);
    assert!(root_move.is_some(), "Root move should be found in book");

    // Deepen cycle 2: Frontier search & expansion
    let _added_2 = book.deepen_book(
        2,
        std::time::Duration::from_millis(60),
        10,
        &legal_lookup,
        &zobrist_constants,
    );

    let count_2 = book.entries.len();
    assert!(count_2 >= count_1, "Book should grow or refine on cycle 2");

    // Clean up temporary test file
    let _ = std::fs::remove_file(&temp_book_path);
}

#[test]
fn test_systematic_egtb_tier9_and_tier8() {
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let temp_egtb_path = std::env::temp_dir().join(format!("test_egtb_{}.bin", rand::random::<u64>()));
    let mut egtb = uttt::egtb::EgtbDatabase::with_path(&temp_egtb_path);

    // 1. Generate Tier 9 (all 9 subboards resolved)
    let added_t9 = egtb.generate_tier_9(&zobrist_constants).expect("Tier 9 generation failed");
    assert!(added_t9 > 500, "Tier 9 should generate hundreds of canonical terminal states");

    // 2. Generate Tier 8 (8 resolved, 1 active)
    let added_t8 = egtb.generate_tier_8(20, &legal_lookup, &zobrist_constants).expect("Tier 8 generation failed");
    assert!(added_t8 > 0, "Tier 8 should generate and solve active subboard positions");

    // 3. Probe tablebase on a sample Tier 9 board
    let mut board = UltimateTicTacToeBoard::default();
    board.x_supers = 7; // subboards 0, 1, 2 won by X (horizontal line)
    for i in 0..9 {
        board.xbb[i] = CANONICAL_X_WIN;
    }
    board.last_super_index = FREE_CHOICE_INDEX;

    let probe_res = egtb.probe(&board, &zobrist_constants);
    assert_eq!(probe_res, Some(GameResult::P1Win), "Tablebase probe must identify proven X win");

    // Clean up temporary test file
    let _ = std::fs::remove_file(&temp_egtb_path);
}


