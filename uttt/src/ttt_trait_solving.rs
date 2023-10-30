use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use std::collections::HashMap;

// tictactoe constants
const TTT_BOARD_WIDTH: usize = 3;
const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
const MOVE_ORDERING: [u8; 9] = [4, 0, 2, 6, 3, 5, 8, 1, 7];
const CAT: u16 = (1 << 9) - 1;

pub const REACHABLE_BOARD_COUNT: usize = 18_753; // At most 3**9=19683 boards should exist
                                                 // 18_753 is the number of tic tac toe boards reachable from the initial position without
                                                 //  requiring that players alternate moves (so x might move twice, then O 3 times.

#[derive(Debug)]
pub struct TicTacToeEngine {
    /// On approaching this project again, it feels like it would be useful to have some sort of
    /// an object to store state related to game search, especially when it comes to handling
    /// stuff like converting to and from IDs, storing computed search results (which will be
    /// more relevant for ultimate tic tac toe since it's actually mildly complicated), and
    /// maybe other little bits of state.
    ///
    boards: [TicTacToeBoard; REACHABLE_BOARD_COUNT],
    board_to_index: HashMap<TicTacToeBoard, u16>,
    board_transitions_x_to_move: [[TicTacToeBoardId; 9]; REACHABLE_BOARD_COUNT],
    board_transitions_o_to_move: [[TicTacToeBoardId; 9]; REACHABLE_BOARD_COUNT],

    board_distance_to_x_win: [u8; REACHABLE_BOARD_COUNT],
    board_distance_to_o_win: [u8; REACHABLE_BOARD_COUNT],
    board_minimax_value_depth_o_to_move: [(i8, u8); REACHABLE_BOARD_COUNT],
    board_minimax_value_depth_x_to_move: [(i8, u8); REACHABLE_BOARD_COUNT],
}

impl Default for TicTacToeEngine {
    fn default() -> Self {
        TicTacToeEngine::new()
    }
}

impl TicTacToeEngine {
    pub fn get_board(&self, board_id: &TicTacToeBoardId) -> &TicTacToeBoard {
        self.boards.get(board_id.board_id as usize).unwrap()
    }

    pub fn new() -> Self {
        // Might be better to use the builder pattern, but I'm going to do it this way!
        eprintln!("generating boards!");
        let mut board_to_index = HashMap::with_capacity(2 * REACHABLE_BOARD_COUNT);
        // 2*REACHABLE_BOARD_COUNT might stop a resize or two, but seems like a good bet.
        let mut numbered_boards: [TicTacToeBoard; REACHABLE_BOARD_COUNT] =
            [TicTacToeBoard::default(); REACHABLE_BOARD_COUNT];
        let mut x_transitions = [[TicTacToeBoardId::default(); 9]; REACHABLE_BOARD_COUNT];
        let mut o_transitions = [[TicTacToeBoardId::default(); 9]; REACHABLE_BOARD_COUNT];
        eprintln!("allocated boards!");

        let init_board = TicTacToeBoard { xbb: 0, obb: 0 };
        let mut board_index_to_expand = 0usize;
        let mut board_index_to_write = 1usize;
        numbered_boards[board_index_to_expand] = init_board;
        for board_index_to_expand in 0..REACHABLE_BOARD_COUNT {
            let board: TicTacToeBoard = numbered_boards[board_index_to_expand];
            if board.get_win_state() == GameResult::Unresolved {
                for x_to_move in &[true, false] {
                    let legal_moves = board.get_legal_moves_no_lookup(*x_to_move);
                    for move_index in 0..legal_moves.move_count {
                        let a_move: GameMove = legal_moves.moves[move_index as usize];
                        let (_unmove, new_board) =
                            board.make_move_const(a_move, &TicTacToeZobrist {});
                        // let board_int = new_board.to_int();
                        let board_index = board_to_index.entry(new_board).or_insert_with(|| {
                            let old_index = board_index_to_write;
                            numbered_boards[board_index_to_write] = new_board;
                            board_index_to_write += 1;
                            old_index as u16
                        });

                        if *x_to_move {
                            x_transitions[board_index_to_expand][a_move.move_index as usize] =
                                TicTacToeBoardId {
                                    board_id: *board_index,
                                };
                        } else {
                            o_transitions[board_index_to_expand][a_move.move_index as usize] =
                                TicTacToeBoardId {
                                    board_id: *board_index,
                                };
                        }
                    }
                }
            }
        }

        let mut board_distance_to_x_win = [u8::MAX; REACHABLE_BOARD_COUNT];
        let mut board_distance_to_o_win = [u8::MAX; REACHABLE_BOARD_COUNT];
        let mut board_minimax_value_depth_x_to_move = [(i8::MIN, u8::MAX); REACHABLE_BOARD_COUNT];
        let mut board_minimax_value_depth_o_to_move = [(i8::MAX, u8::MAX); REACHABLE_BOARD_COUNT];

        for board_index in (0..REACHABLE_BOARD_COUNT).rev() {
            let board_id = TicTacToeBoardId {
                board_id: board_index as u16,
            };
            let board: TicTacToeBoard = numbered_boards[board_index];
            let board_win_state = board.get_win_state();
            if board_win_state == GameResult::P1Win {
                board_distance_to_x_win[board_index] = 0;
                board_minimax_value_depth_o_to_move[board_index] = (i8::MAX, 0);
                board_minimax_value_depth_x_to_move[board_index] = (i8::MAX, 0);
            } else if board_win_state == GameResult::P2Win {
                board_distance_to_o_win[board_index] = 0;
                board_minimax_value_depth_o_to_move[board_index] = (i8::MIN, 0);
                board_minimax_value_depth_x_to_move[board_index] = (i8::MIN, 0);
            } else if board_win_state == GameResult::Tie {
                board_minimax_value_depth_o_to_move[board_index] = (0, 0);
                board_minimax_value_depth_x_to_move[board_index] = (0, 0);
            } else {
                // use pre-computed transitions to look over successors
                let successors_x_to_move = x_transitions[board_index]
                    .iter()
                    .filter(|each_successor| (each_successor.board_id != 0));

                let this_distance_to_x_win: u8 = 1 + successors_x_to_move
                    .clone()
                    .map(|each_successor| board_distance_to_x_win[each_successor.board_id as usize])
                    .min()
                    .unwrap()
                    .min(254);
                board_distance_to_x_win[board_index] = this_distance_to_x_win;

                let (minimax_val, minimax_depth): (i8, u8) = successors_x_to_move
                    .clone()
                    .map(|each_successor| {
                        board_minimax_value_depth_o_to_move[each_successor.board_id as usize]
                    })
                    .max_by_key(|(value, depth)| (*value, 9 - depth))
                    .unwrap();
                board_minimax_value_depth_x_to_move[board_index] = (minimax_val, 1 + minimax_depth);

                let successors_o_to_move = o_transitions[board_index]
                    .iter()
                    .filter(|each_successor| (each_successor.board_id != 0));

                let this_distance_to_o_win: u8 = 1 + successors_o_to_move
                    .clone()
                    .map(|each_successor| board_distance_to_o_win[each_successor.board_id as usize])
                    .min()
                    .unwrap()
                    .min(254);
                board_distance_to_o_win[board_index] = this_distance_to_o_win;

                let (minimax_val, minimax_depth): (i8, u8) = successors_o_to_move
                    .clone()
                    .map(|each_successor| {
                        board_minimax_value_depth_x_to_move[each_successor.board_id as usize]
                    })
                    .min_by_key(|(value, depth)| (*value, 9 - *depth))
                    .unwrap();
                board_minimax_value_depth_o_to_move[board_index] = (minimax_val, 1 + minimax_depth);
            }
        }

        TicTacToeEngine {
            boards: numbered_boards,
            board_to_index,
            board_transitions_x_to_move: x_transitions,
            board_transitions_o_to_move: o_transitions,
            board_distance_to_x_win,
            board_distance_to_o_win,
            board_minimax_value_depth_o_to_move,
            board_minimax_value_depth_x_to_move,
        }
    }

    pub fn get_move_ordering_array(&self) -> Vec<Vec<GameMove>> {
        let mut move_vec: Vec<Vec<GameMove>> = vec![Vec::with_capacity(0); 524_288];
        for (number, each_board) in self.boards.iter().enumerate() {
            let TicTacToeBoard { xbb, obb } = each_board;
            for x_to_move in &[true, false] {
                let board = TicTacToeState {
                    xbb: *xbb,
                    obb: *obb,
                    x_to_move: *x_to_move,
                };
                let big_index = board.to_index();

                let successor_ids = if *x_to_move {
                    self.board_transitions_x_to_move[number]
                } else {
                    self.board_transitions_o_to_move[number]
                };

                let mut valid_successors: Vec<(usize, &TicTacToeBoardId)> = successor_ids
                    .iter()
                    .enumerate()
                    .filter(|each_successor| {
                        let (_index, id) = each_successor;
                        TicTacToeBoardId { board_id: 0 }.ne(id)
                    })
                    .collect();

                if *x_to_move {
                    valid_successors.sort_by_key(|(_move_number, board_id)| {
                        let successor_id = board_id.board_id as usize;
                        let (val, _val_depth) =
                            self.board_minimax_value_depth_x_to_move[successor_id];
                        let threat_depth = self.board_distance_to_x_win[successor_id];
                        let defense_depth = self.board_distance_to_o_win[successor_id];
                        (-(val as i16), threat_depth, -(defense_depth as i16))
                    });
                } else {
                    valid_successors.sort_by_key(|(move_number, board_id)| {
                        let successor_id = board_id.board_id as usize;
                        let (val, val_depth) =
                            self.board_minimax_value_depth_o_to_move[successor_id];
                        let threat_depth = self.board_distance_to_o_win[successor_id];
                        let defense_depth = self.board_distance_to_x_win[successor_id];
                        (-(val as i16), threat_depth, -(defense_depth as i16))
                    });
                }

                move_vec[big_index as usize] = valid_successors
                    .iter()
                    .map(|(move_number, _board_id)| GameMove {
                        move_index: *move_number as u8,
                        x_to_move: *x_to_move,
                    })
                    .collect();
            }
        }
        move_vec
    }
}

pub const BB_STATE_CHECK: [GameResult; 512] = TicTacToeBoard::make_bb_lookup();

#[derive(Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct TicTacToeTerminal {
    pub terminal_value: i8,
    pub depth_to_terminal: u8,
}

impl Default for TicTacToeTerminal {
    fn default() -> TicTacToeTerminal {
        TicTacToeTerminal {
            terminal_value: 0,
            depth_to_terminal: 9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TicTacToeState {
    pub xbb: u16,
    pub obb: u16,
    pub x_to_move: bool,
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct TicTacToeBoard {
    pub xbb: u16,
    pub obb: u16,
}
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct TicTacToeBoardId {
    pub board_id: u16,
}
impl TicTacToeBoardId {
    pub const fn default() -> TicTacToeBoardId {
        TicTacToeBoardId { board_id: 0 }
    }
}

#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct OrderedMoves {
    move_count: u8,
    moves: [GameMove; 9],
}
impl OrderedMoves {
    pub const fn default() -> OrderedMoves {
        let default_move = GameMove {
            x_to_move: true,
            move_index: 0,
        };
        OrderedMoves {
            move_count: 0,
            moves: [default_move; 9],
        }
    }
}

pub struct TicTacToeUnmove {
    pub x_to_move: bool,
    pub move_index: u8,
}

pub struct TicTacToeZobrist {
    //I'm not ACTUALLY implementing Zobrist constants for normal tictactoe...
    // I have to create this to match the make_move API though, so here we are!
}

impl TicTacToeState {
    pub fn to_index(&self) -> u32 {
        (u32::from(self.xbb) << 10)
            | (u32::from(self.obb) << 1)
            | (if self.x_to_move { 1u32 } else { 0u32 })
    }

    fn get_legal_moves_no_lookup(&self) -> Vec<GameMove> {
        let occupied_bb = self.xbb | self.obb;

        MOVE_ORDERING
            .iter()
            .filter_map(|a_move_index| match occupied_bb & (1u16 << a_move_index) {
                0 => Some(GameMove {
                    x_to_move: self.x_to_move,
                    move_index: *a_move_index,
                }),
                _nonzero => None,
            })
            .collect()
    }
}

impl TicTacToeBoard {
    const fn default() -> TicTacToeBoard {
        TicTacToeBoard { xbb: 0, obb: 0 }
    }

    pub fn display(&self) -> String {
        let mut rows = Vec::new();
        for row in 0..TTT_BOARD_WIDTH {
            let mut row_rep = String::new();
            for col in 0..TTT_BOARD_WIDTH {
                let mask = 1 << (col + TTT_BOARD_WIDTH * row);
                if self.xbb & mask != 0 {
                    row_rep.push('X');
                } else if self.obb & mask != 0 {
                    row_rep.push('O');
                } else {
                    row_rep.push('.');
                }
            }
            rows.push(row_rep);
        }
        rows.join("\n")
    }

    pub const fn get_win_state(&self) -> GameResult {
        // Forgive the weird casting here, it's due to const fn. See:
        //   https://stackoverflow.com/questions/60125657/rust-cant-use-enum-in-const-fn

        if (BB_STATE_CHECK[self.xbb as usize] as isize) == (GameResult::P1Win as isize) {
            GameResult::P1Win
        } else if (BB_STATE_CHECK[self.obb as usize] as isize) == (GameResult::P1Win as isize) {
            GameResult::P2Win
        } else if (self.xbb | self.obb) == CAT {
            GameResult::Tie
        } else {
            GameResult::Unresolved
        }
    }

    pub const fn get_win_state_alternate(&self) -> GameResult {
        // check for X or O win patterns, check for a filled board, or return unresolved
        let mut mask_index = 0;
        while mask_index < RESULT_MASKS.len() {
            if (self.xbb & RESULT_MASKS[mask_index]) == RESULT_MASKS[mask_index] {
                return GameResult::P1Win;
            } else if (self.obb & RESULT_MASKS[mask_index]) == RESULT_MASKS[mask_index] {
                return GameResult::P2Win;
            }
            mask_index += 1;
        }
        if (self.xbb | self.obb) == CAT {
            GameResult::Tie
        } else {
            GameResult::Unresolved
        }
    }

    pub const fn get_legal_moves_no_lookup(&self, x_to_move: bool) -> OrderedMoves {
        let occupied_bb = self.xbb | self.obb;

        // let mut move_order_copy = MOVE_ORDERING;
        let mut ordered_moves = OrderedMoves::default();
        // let mut moves: Vec<GameMove> = Vec::new();

        let mut move_index = 0u8;
        let mut assigned_moves = 0;
        while (move_index as usize) < MOVE_ORDERING.len() {
            if occupied_bb & (1u16 << move_index) == 0 {
                let new_move = GameMove {
                    x_to_move,
                    move_index,
                };
                ordered_moves.moves[assigned_moves] = new_move;
                ordered_moves.move_count += 1;
                assigned_moves += 1;
            }
            move_index += 1;
        }

        ordered_moves
    }

    pub const fn to_int(&self) -> usize {
        // xbb and obb are bitboards with 9 bits
        ((self.xbb as usize) << 9) | (self.obb as usize)
    }

    pub const fn make_move_const(
        &self,
        a_move: GameMove,
        _zobrist_constants: &TicTacToeZobrist,
    ) -> (TicTacToeUnmove, TicTacToeBoard) {
        let mut board_copy = *self;
        let unmove = TicTacToeUnmove {
            x_to_move: a_move.x_to_move,
            move_index: a_move.move_index,
        };
        match a_move {
            GameMove {
                x_to_move: true,
                move_index,
            } => {
                board_copy.xbb ^= 1u16 << move_index;
            }
            GameMove {
                x_to_move: false,
                move_index,
            } => {
                board_copy.obb ^= 1u16 << move_index;
            }
        }
        (unmove, board_copy)
    }

    pub const fn make_bb_lookup() -> [GameResult; 512] {
        let mut bb = 0u16;
        let mut results = [GameResult::Unresolved; 512];
        while bb < 512 {
            let mut mask_index = 0;
            while mask_index < RESULT_MASKS.len() {
                if bb & RESULT_MASKS[mask_index] == RESULT_MASKS[mask_index] {
                    results[bb as usize] = GameResult::P1Win;
                }
                mask_index += 1;
            }
            bb += 1;
        }

        results
    }
}

impl AlphaBetaCompatible<u32, TicTacToeUnmove, TicTacToeZobrist> for TicTacToeState {
    fn hash_val(&self) -> u32 {
        let base_hash = (u32::from(self.xbb)) << 16 | u32::from(self.obb);

        if self.x_to_move {
            base_hash
        } else {
            !base_hash
        }
    }

    fn get_win_state(&self) -> GameResult {
        // check for X or O win patterns, check for a filled board, or return unresolved
        if RESULT_MASKS.iter().any(|x| (x & self.xbb).eq(x)) {
            GameResult::P1Win
        } else if RESULT_MASKS.iter().any(|x| (x & self.obb).eq(x)) {
            GameResult::P2Win
        } else if (self.xbb | self.obb) == CAT {
            GameResult::Tie
        } else {
            GameResult::Unresolved
        }
    }

    fn x_to_move(&self) -> bool {
        self.x_to_move
    }

    fn make_move(
        &mut self,
        a_move: GameMove,
        _zobrist_constants: &TicTacToeZobrist,
    ) -> TicTacToeUnmove {
        //            pub x_to_move: bool,
        //            pub move_index: u8,
        let unmove = TicTacToeUnmove {
            x_to_move: self.x_to_move,
            move_index: a_move.move_index,
        };

        match a_move {
            GameMove {
                x_to_move: true,
                move_index,
            } => {
                self.xbb ^= 1u16 << move_index;
                self.x_to_move = false;
            }
            GameMove {
                x_to_move: false,
                move_index,
            } => {
                self.obb ^= 1u16 << move_index;
                self.x_to_move = true;
            }
        }

        unmove
    }

    fn unmake_move(&mut self, unmove: TicTacToeUnmove) {
        match unmove {
            TicTacToeUnmove {
                x_to_move: true,
                move_index,
            } => {
                self.xbb ^= 1u16 << move_index;
                self.x_to_move = true;
            }
            TicTacToeUnmove {
                x_to_move: false,
                move_index,
            } => {
                self.obb ^= 1u16 << move_index;
                self.x_to_move = false;
            }
        }
    }

    fn score(&self) -> i8 {
        match self.get_win_state() {
            GameResult::P1Win => i8::MAX,
            GameResult::P2Win => i8::MIN,
            GameResult::Tie => 0,
            GameResult::Unresolved if self.x_to_move => 1,
            GameResult::Unresolved => -1,
        }
    }

    fn get_legal_moves(&self, _legal_lookup: &[Vec<GameMove>]) -> Vec<GameMove> {
        //TODO: This should probably use the legal lookup directly
        let board_state = TicTacToeState {
            xbb: self.xbb,
            obb: self.obb,
            x_to_move: self.x_to_move,
        };
        let board_index = board_state.to_index();
        _legal_lookup[board_index as usize].clone()
    }

    fn display(&self) -> String {
        //        let mut board_rep = String::new();
        let board = TicTacToeBoard {
            xbb: self.xbb,
            obb: self.obb,
        };
        board.display()
    }

    fn is_valid(&self) -> bool {
        // there are more ways to invalidate a board than are checked here
        (self.obb & self.xbb) == 0
    }
}
