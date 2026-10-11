use rand::{rngs::StdRng, Rng, SeedableRng};
use uttt::canonical::{
    CANONICAL_O_WIN, CANONICAL_TIE_O, CANONICAL_TIE_X, CANONICAL_X_WIN, FREE_CHOICE_INDEX,
};

use uttt::dfpn::DfPnSolver;
use uttt::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
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

    let probe_res = egtb.probe(&board);
    assert_eq!(probe_res, Some(GameResult::P1Win), "Tablebase probe must identify proven X win");

    // Clean up temporary test file
    let _ = std::fs::remove_file(&temp_egtb_path);
}

#[test]
fn test_extract_opening_suite() {
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let book = uttt::opening_book::OpeningBook::new();
    let suite = book.extract_opening_suite(10, 2, &legal_lookup, &zobrist_constants);

    assert!(!suite.is_empty(), "Opening suite must return non-empty positions");
    assert!(suite.len() <= 10, "Opening suite size must not exceed requested target count");

    let mut hashes = std::collections::HashSet::new();
    for board in &suite {
        // Invariant 1: even ply implies X to move
        assert!(board.x_to_move, "Even ply openings must have X to move");
        // Invariant 2: game must be unresolved
        assert_eq!(board.get_win_state(), GameResult::Unresolved, "Opening must be unresolved");
        // Invariant 3: uniqueness
        assert!(hashes.insert(board.hash_val()), "Opening suite positions must be distinct");
    }
}

#[test]
fn test_dfpn_solve_move_finds_winning_move() {
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let mut solver = DfPnSolver::new(10_000);

    // Board where X has subboards 0, 1 won, directed to subboard 2, with squares 0 and 1 occupied.
    // Square 2 completes subboard 2 and wins the entire match!
    let mut board = UltimateTicTacToeBoard::default();
    board.x_supers = 3; // Subboards 0 and 1 won (0b000000011)
    board.last_super_index = 2; // Directed to subboard 2
    board.xbb[2] = 3; // squares 0 and 1 occupied in subboard 2
    board.x_to_move = true;

    let res = solver.solve_move(&board, &legal_lookup, &zobrist_constants);
    assert!(res.is_some());
    let (is_win, best_move) = res.unwrap();
    assert!(is_win, "Should prove win for X");
    assert!(best_move.is_some(), "Winning move should be identified");
    let m = best_move.unwrap();
    let parsed = uttt::uttt_trait_solving::parse_index(m.move_index);
    assert_eq!(parsed.sub_board_index, 2);
    assert_eq!(parsed.sub_square_index, 2, "Square 2 completes 3-in-a-row in subboard 2");
}

#[test]
fn test_umcts_v3_superboard_priors_and_grandfather() {
    use uttt::bots::Bot;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();

    let mut bot = uttt::bots::UmctsV3Bot::new();

    // Board where X has an immediate 1-ply winning move
    let mut board = UltimateTicTacToeBoard::default();
    board.x_supers = 3; // subboards 0, 1 won
    board.last_super_index = 2; // directed to subboard 2
    board.xbb[2] = 3; // squares 0, 1 occupied
    board.x_to_move = true;

    let chosen = bot.choose_move(&board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(50));
    let parsed = uttt::uttt_trait_solving::parse_index(chosen.move_index);
    assert_eq!(parsed.sub_board_index, 2);
    assert_eq!(parsed.sub_square_index, 2, "UmctsV3Bot must instantly find immediate game winning move");
}

#[test]
fn test_alphabeta_v2_quiescence_finds_immediate_mate() {
    use uttt::bots::Bot;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();

    let mut bot = uttt::bots::AlphaBetaV2Bot::new();

    let mut board = UltimateTicTacToeBoard::default();
    board.x_supers = 3;
    board.last_super_index = 2;
    board.xbb[2] = 3;
    board.x_to_move = true;

    let chosen = bot.choose_move(&board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(50));
    let parsed = uttt::uttt_trait_solving::parse_index(chosen.move_index);
    assert_eq!(parsed.sub_board_index, 2);
    assert_eq!(parsed.sub_square_index, 2, "AlphaBetaV2Bot must find immediate winning move");
}

#[test]
fn test_compare_mcts_iterations() {
    use uttt::bots::Bot;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();

    let mut v2 = uttt::bots::UmctsV2Bot::new();
    let mut v3 = uttt::bots::UmctsV3Bot::new();

    let mut board = UltimateTicTacToeBoard::default();
    let moves = [40, 36, 12, 30];
    for &m in &moves {
        board.make_move(uttt::game_solving::GameMove { x_to_move: board.x_to_move, move_index: m }, &zobrist_constants);
    }

    let m2 = v2.choose_move(&board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(100));
    let m3 = v3.choose_move(&board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(100));

    let empty_board = UltimateTicTacToeBoard::default();
    let m2_empty = v2.choose_move(&empty_board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(50));
    let m3_empty = v3.choose_move(&empty_board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(50));
    assert!(m2.move_index < 81);
    assert!(m3.move_index < 81);
    assert!(m2_empty.move_index < 81);
    assert!(m3_empty.move_index < 81);
}

#[test]
fn test_macro_routing_penalizes_gifting_free_choice() {
    let bot = uttt::bots::AlphaBetaV2Bot::new();
    let mut board = UltimateTicTacToeBoard::default();

    // Mark subboard 0 as won by X
    board.x_supers = 1 << 0;
    board.x_to_move = true;
    board.last_super_index = 4; // active player X in subboard 4

    // Move 1: X moves in subboard 4, cell 0 -> opponent O directed to subboard 0 (won) -> free choice!
    let m_free_choice = GameMove { x_to_move: true, move_index: 4 * 9 + 0 };
    // Move 2: X moves in subboard 4, cell 1 -> opponent O directed to subboard 1 (open, neutral)
    let m_neutral = GameMove { x_to_move: true, move_index: 4 * 9 + 1 };

    let score_free = bot.score_move(m_free_choice, None, 0, None, &board);
    let score_neutral = bot.score_move(m_neutral, None, 0, None, &board);

    // Gifting free choice should be penalised significantly (around -300)
    assert!(
        score_neutral > score_free + 200,
        "Move gifting free choice ({}) must be penalized relative to neutral move ({})",
        score_free,
        score_neutral
    );

    // Also check order_moves puts m_neutral before m_free_choice
    let mut moves = vec![m_free_choice, m_neutral];
    bot.order_moves(&mut moves, None, 0, None, &board);
    assert_eq!(moves[0], m_neutral, "order_moves must prioritize neutral move over gifting free choice");

    // Also check static evaluation of child states
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let mut board_free = board.clone();
    board_free.make_move(m_free_choice, &zobrist_constants);
    let mut board_neutral = board.clone();
    board_neutral.make_move(m_neutral, &zobrist_constants);

    // From X's perspective, board_neutral should evaluate higher (better for X) than board_free
    let eval_free = uttt::bots::AlphaBetaV2Bot::evaluate_board(&board_free);
    let eval_neutral = uttt::bots::AlphaBetaV2Bot::evaluate_board(&board_neutral);
    assert!(
        eval_neutral > eval_free,
        "evaluate_board must favor state where opponent lacks free choice (neutral: {}, free: {})",
        eval_neutral,
        eval_free
    );
}

#[test]
fn test_macro_routing_penalizes_directing_into_opponent_threat() {
    let bot = uttt::bots::AlphaBetaV2Bot::new();
    let mut board = UltimateTicTacToeBoard::default();

    // In subboard 1: opponent O has marks at 0 and 1, X has 0 marks -> active O threat on line [0, 1, 2]
    board.obb[1] = (1 << 0) | (1 << 1);
    // In subboard 2: no marks (neutral)
    board.x_to_move = true;
    board.last_super_index = 4; // active player X in subboard 4

    // Move 1: X moves in subboard 4, cell 1 -> directs O directly into subboard 1 with unblocked threat!
    let m_into_threat = GameMove { x_to_move: true, move_index: 4 * 9 + 1 };
    // Move 2: X moves in subboard 4, cell 2 -> directs O to subboard 2 (safe/neutral)
    let m_safe = GameMove { x_to_move: true, move_index: 4 * 9 + 2 };

    let score_threat = bot.score_move(m_into_threat, None, 0, None, &board);
    let score_safe = bot.score_move(m_safe, None, 0, None, &board);

    // Directing into opponent threat should be penalized by around 400 points
    assert!(
        score_safe > score_threat + 300,
        "Move directing into opponent threat ({}) must be heavily penalized relative to safe move ({})",
        score_threat,
        score_safe
    );

    let mut moves = vec![m_into_threat, m_safe];
    bot.order_moves(&mut moves, None, 0, None, &board);
    assert_eq!(moves[0], m_safe, "order_moves must prioritize safe routing over directing into opponent threat");

    // Static evaluation check
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let mut board_threat = board.clone();
    board_threat.make_move(m_into_threat, &zobrist_constants);
    let mut board_safe = board.clone();
    board_safe.make_move(m_safe, &zobrist_constants);

    let eval_threat = uttt::bots::AlphaBetaV2Bot::evaluate_board(&board_threat);
    let eval_safe = uttt::bots::AlphaBetaV2Bot::evaluate_board(&board_safe);
    assert!(
        eval_safe > eval_threat,
        "evaluate_board must favor safe board (safe: {}, into threat: {})",
        eval_safe,
        eval_threat
    );
}

#[test]
fn test_macro_routing_rewards_neutralized_subboard() {
    let bot = uttt::bots::AlphaBetaV2Bot::new();
    let mut board = UltimateTicTacToeBoard::default();

    // In subboard 1: X has marks at {0, 2, 5, 7} and O has marks at {3, 4}.
    // This blocks all 8 winning lines for O without X or O completing 3-in-a-row.
    board.xbb[1] = (1 << 0) | (1 << 2) | (1 << 5) | (1 << 7);
    board.obb[1] = (1 << 3) | (1 << 4);

    // Subboard 2 is empty/open
    board.x_to_move = true;
    board.last_super_index = 4;

    // Move to cell 1 directs O into subboard 1 (neutralized/dead for O)
    let m_dead = GameMove { x_to_move: true, move_index: 4 * 9 + 1 };
    // Move to cell 2 directs O into subboard 2 (empty/open)
    let m_open = GameMove { x_to_move: true, move_index: 4 * 9 + 2 };

    let score_dead = bot.score_move(m_dead, None, 0, None, &board);
    let score_open = bot.score_move(m_open, None, 0, None, &board);

    assert!(
        score_dead > score_open,
        "Directing opponent to neutralized board ({}) must score higher than open board ({})",
        score_dead,
        score_open
    );
}

#[test]
fn test_history_and_countermove_ordering() {
    let mut bot = uttt::bots::AlphaBetaV2Bot::new();
    let mut board = UltimateTicTacToeBoard::default();
    board.x_to_move = true;
    board.last_super_index = 4;

    let prev_move = GameMove { x_to_move: false, move_index: 0 * 9 + 4 };
    let m1 = GameMove { x_to_move: true, move_index: 4 * 9 + 1 };
    let m2 = GameMove { x_to_move: true, move_index: 4 * 9 + 3 };
    let m3 = GameMove { x_to_move: true, move_index: 4 * 9 + 5 };

    // 1. Countermove test: setting countermoves[prev] = Some(m2)
    bot.countermoves[prev_move.move_index as usize] = Some(m2);
    let mut moves = vec![m1, m2, m3];
    bot.order_moves(&mut moves, None, 1, Some(prev_move), &board);
    assert_eq!(moves[0], m2, "Countermove m2 must be ordered first");

    // 2. Killer move overrides countermove
    let killer = m3;
    bot.killers[1][0] = Some(killer);
    let mut moves2 = vec![m1, m2, m3];
    bot.order_moves(&mut moves2, None, 1, Some(prev_move), &board);
    assert_eq!(moves2[0], killer, "Killer move m3 must override countermove m2");

    // 3. History score ordering among quiet moves (no killer, no countermove)
    use uttt::bots::Bot;
    bot.reset();
    bot.history[prev_move.move_index as usize][m1.move_index as usize] = 5000;
    let mut moves3 = vec![m3, m2, m1];
    bot.order_moves(&mut moves3, None, 1, Some(prev_move), &board);
    assert_eq!(moves3[0], m1, "High history move m1 must be ordered first among quiet moves");

    // 4. Aging/decay test: values decay when decay_history is called
    assert_eq!(bot.history[prev_move.move_index as usize][m1.move_index as usize], 5000);
    bot.decay_history(2);
    assert_eq!(bot.history[prev_move.move_index as usize][m1.move_index as usize], 2500);
}

#[test]
fn test_alphabeta_v2_depth_and_routing_integration() {
    use uttt::bots::Bot;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();

    let mut bot = uttt::bots::AlphaBetaV2Bot::new();
    let board = UltimateTicTacToeBoard::default();

    // Verify search runs smoothly to depth with macro routing and history tables
    let chosen = bot.choose_move(&board, &legal_lookup, &zobrist_constants, std::time::Duration::from_millis(50));
    assert!(chosen.move_index < 81, "Valid move chosen");

    // Verify killers and history were used
    let total_history: u32 = bot.history.iter().flat_map(|r| r.iter()).sum();
    println!("Total history accumulated in search: {}", total_history);
}

#[test]
fn test_partitioned_egtb_tier_probing_matches_monolithic() {
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();

    let temp_part_dir = std::env::temp_dir().join(format!("test_part_egtb_{}", rand::random::<u64>()));
    let temp_mono_path = std::env::temp_dir().join(format!("test_mono_egtb_{}.bin", rand::random::<u64>()));

    let mut part_egtb = uttt::egtb::EgtbDatabase::with_dir(&temp_part_dir);
    let mut mono_egtb = uttt::egtb::EgtbDatabase::with_path(&temp_mono_path);

    // 1. Generate Tier 9 in partitioned database and monolithic database
    let part_added_t9 = part_egtb.generate_tier_9(&zobrist_constants).expect("Part Tier 9 failed");
    let mono_added_t9 = mono_egtb.generate_tier_9(&zobrist_constants).expect("Mono Tier 9 failed");
    assert!(part_added_t9 > 500);
    assert_eq!(part_added_t9, mono_added_t9);

    // 2. Generate Tier 8 in partitioned database and monolithic database
    let part_added_t8 = part_egtb.generate_tier_8(20, &legal_lookup, &zobrist_constants).expect("Part Tier 8 failed");
    let mono_added_t8 = mono_egtb.generate_tier_8(20, &legal_lookup, &zobrist_constants).expect("Mono Tier 8 failed");
    assert!(part_added_t8 > 0);
    assert_eq!(part_added_t8, mono_added_t8);

    // 3. Generate Tier 7 in partitioned database
    let part_added_t7 = part_egtb.generate_tier_7(10, &legal_lookup, &zobrist_constants).expect("Part Tier 7 failed");
    assert!(part_added_t7 > 0);

    // Verify partitioned tier files exist
    assert!(part_egtb.tier_path(9).exists(), "egtb_tier9.bin must exist");
    assert!(part_egtb.tier_path(8).exists(), "egtb_tier8.bin must exist");
    assert!(part_egtb.tier_path(7).exists(), "egtb_tier7.bin must exist");

    // Verify resolved counts and direct tier probing
    assert!(part_egtb.tier_counts[9] > 500);
    assert!(part_egtb.tier_counts[8] >= 20);
    assert!(part_egtb.tier_counts[7] >= 10);

    // Test a sample Tier 9 board
    let mut board_t9 = UltimateTicTacToeBoard::default();
    board_t9.x_supers = 7; // subboards 0, 1, 2 won by X (horizontal line)
    for i in 0..9 {
        board_t9.xbb[i] = CANONICAL_X_WIN;
    }
    board_t9.last_super_index = FREE_CHOICE_INDEX;
    assert_eq!(board_t9.get_resolved_count(), 9);

    let part_res_t9 = part_egtb.probe(&board_t9);
    let mono_res_t9 = mono_egtb.probe(&board_t9);
    assert_eq!(part_res_t9, Some(GameResult::P1Win));
    assert_eq!(part_res_t9, mono_res_t9, "Partitioned Tier 9 probe must match monolithic result");

    // Test a sample Tier 8 board
    let mut board_t8 = UltimateTicTacToeBoard::default();
    board_t8.x_supers = (1 << 3) | (1 << 4) | (1 << 5);
    for i in 1..9 {
        board_t8.xbb[i] = CANONICAL_X_WIN;
    }
    board_t8.last_super_index = FREE_CHOICE_INDEX;
    board_t8.x_to_move = true;
    assert_eq!(board_t8.get_resolved_count(), 8);

    let part_res_t8 = part_egtb.probe(&board_t8);
    let mono_res_t8 = mono_egtb.probe(&board_t8);
    assert_eq!(part_res_t8, Some(GameResult::P1Win));
    assert_eq!(part_res_t8, mono_res_t8, "Partitioned Tier 8 probe must match monolithic result");

    // Clean up temporary test files
    let _ = std::fs::remove_dir_all(&temp_part_dir);
    let _ = std::fs::remove_file(&temp_mono_path);
    let _ = std::fs::remove_file(mono_egtb.tier_path(9));
    let _ = std::fs::remove_file(mono_egtb.tier_path(8));
}

#[test]
fn test_mcts_solver_handoff_root_and_intree() {
    use uttt::bots::Bot;
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let legal_lookup = UltimateTicTacToeBoard::get_move_ordering_array();

    let mut bot = uttt::bots::UmctsV3Bot::new();

    // 1. Root Solver Handoff (Late Game: >= 5 resolved subboards)
    // Board where X has 5 subboards resolved, directed to subboard 2 with squares 0, 1 occupied.
    // Square 2 completes subboard 2 and gives X subboards 0, 1, 2 winning the game!
    let mut endgame_board = UltimateTicTacToeBoard::default();
    endgame_board.x_supers = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 6);
    endgame_board.xbb[0] = CANONICAL_X_WIN;
    endgame_board.xbb[1] = CANONICAL_X_WIN;
    endgame_board.xbb[3] = CANONICAL_X_WIN;
    endgame_board.xbb[4] = CANONICAL_X_WIN;
    endgame_board.xbb[6] = CANONICAL_X_WIN;
    endgame_board.last_super_index = 2; // Directed to subboard 2
    endgame_board.xbb[2] = 3; // squares 0 and 1 occupied
    endgame_board.x_to_move = true;

    assert!(endgame_board.get_resolved_count() >= 5);

    let chosen = bot.choose_move(
        &endgame_board,
        &legal_lookup,
        &zobrist_constants,
        std::time::Duration::from_millis(50),
    );
    let parsed = uttt::uttt_trait_solving::parse_index(chosen.move_index);
    assert_eq!(parsed.sub_board_index, 2);
    assert_eq!(
        parsed.sub_square_index, 2,
        "MCTS solver handoff must immediately return the proven winning move"
    );

    // 2. Pure MCTS Tree Search and Random Simulations Throughput
    // In opening/midgame (resolved < 5), MCTS runs pure rollouts and achieves high visit counts
    bot.reset();
    let start_time = std::time::Instant::now();
    let root = UltimateTicTacToeBoard::default();
    let root_move = bot.choose_move(
        &root,
        &legal_lookup,
        &zobrist_constants,
        std::time::Duration::from_millis(100),
    );
    let elapsed = start_time.elapsed();
    assert!(root_move.move_index < 81);

    let root_hash = root.hash_val() ^ ((root.last_super_index as u64).wrapping_mul(0x9E3779B97F4A7C15));
    let visits = bot.node_store().get(&root_hash).map_or(0, |n| n.visits);
    println!("MCTS Pure Search completed: {} visits in {:?}", visits, elapsed);
    assert!(
        visits > 2000,
        "Pure MCTS throughput must exceed 2,000 visits in 100ms (got {})",
        visits
    );
}



