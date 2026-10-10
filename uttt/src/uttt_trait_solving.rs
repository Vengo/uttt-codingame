use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::ttt_trait_solving::{TicTacToeEngine, TicTacToeState};
use rand::{rngs, Rng, SeedableRng};
use std::iter::FromIterator;

// tictactoe constants
const RESULT_MASKS: [u16; 8] = [7, 7 << 3, 7 << 6, 73, 73 << 1, 73 << 2, 84, 273];
const CAT: u16 = (1 << 9) - 1;
const ZOBRIST_SEED: u64 = 0x95_62_9A_0C_B8_00_E9_41u64; //generated from random.org
pub const HASH_DEFAULT: u64 = 0x8e_6c_65_b8_2e_fd_36_70u64; //generated from random.org

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct UltimateTicTacToeBoard {
    pub xbb: [u16; 9],
    pub x_supers: u16,
    pub obb: [u16; 9],
    pub o_supers: u16,
    //        cbb: u128,
    //        c_supers: u16,
    pub last_super_index: u8,
    pub x_to_move: bool,
    pub hash_state: u64,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct SubBoardAndSquare {
    pub sub_board_index: usize,
    pub sub_square_index: usize,
}

pub fn parse_index(an_index: u8) -> SubBoardAndSquare {
    // Returns the sub board [0,8] and sub square [0-8]
    let sub_square = (an_index % 9) as usize;
    let sub_board = (an_index / 9) as usize;

    SubBoardAndSquare {
        sub_square_index: sub_square,
        sub_board_index: sub_board,
    }
}

pub fn parse_index_from_super_coords(row_index: u8, col_index: u8) -> u8 {
    let board_row = row_index / 3;
    let subboard_row = row_index % 3;

    let board_col = col_index / 3;
    let subboard_col = col_index % 3;
    // Returns the sub board [0,8] and sub square [0-8]

    let subboard_index = 3 * board_row + board_col;
    let subsquare_index = 3 * subboard_row + subboard_col;

    9 * subboard_index + subsquare_index
}

#[derive(Clone, Copy)]
pub struct UltimateTicTacToeUnmove {
    x_to_move: bool,
    move_index: u8,
    super_unmask: u16,
    last_super: u8,
    hash_reset: u64,
}

pub struct ZobristConstants {
    pub x_constants: [u64; 81],
    pub o_constants: [u64; 81],
    pub x_to_move: u64,
}

impl Default for UltimateTicTacToeBoard {
    fn default() -> UltimateTicTacToeBoard {
        UltimateTicTacToeBoard {
            xbb: [0u16; 9],
            obb: [0u16; 9],
            last_super_index: 9,
            o_supers: 0,
            x_supers: 0,
            x_to_move: true,
            hash_state: HASH_DEFAULT,
        }
    }
}

impl UltimateTicTacToeBoard {
    pub fn get_tied_mask(&self) -> u16 {
        self.xbb
            .iter()
            .zip(self.obb.iter())
            .enumerate()
            .filter_map(|(index, (xbb, obb))| match xbb | obb {
                CAT => Some(1u16 << index),
                _ => None,
            })
            .fold(0u16, |current, next| current | next)
    }

    pub fn get_subboard_moves(
        &self,
        subboard_index: usize,
        legality_lookup: &[Vec<GameMove>],
        side_to_move: bool,
    ) -> Vec<GameMove> {
        let subboard = TicTacToeState {
            xbb: self.xbb[subboard_index],
            obb: self.obb[subboard_index],
            x_to_move: side_to_move,
        };

        let lookup_index = subboard.to_index();
        match legality_lookup.get(lookup_index as usize) {
            Some(move_vec) => move_vec
                .iter()
                .map(
                    |GameMove {
                         x_to_move: _side,
                         move_index: a_move_index,
                     }| GameMove {
                        x_to_move: side_to_move,
                        move_index: a_move_index + 9 * subboard_index as u8,
                    },
                )
                .collect(),
            None => {
                panic!("It shouldn't be possible for subboard moves not to find a move vector")
            }
        }
    }

    pub fn get_move_ordering_array() -> Vec<Vec<GameMove>> {
        eprintln!("making an engine");
        let engine = TicTacToeEngine::new();
        eprintln!("engine made");
        engine.get_move_ordering_array()
    }

    pub fn get_zobrist_constants() -> ZobristConstants {
        // zobrist hashing using a seedable RNG, and a seed from random.org
        let mut seeded = rngs::StdRng::seed_from_u64(ZOBRIST_SEED);
        let x_consts = UltimateTicTacToeBoard::generate_zobrist_random(&mut seeded);
        let o_consts = UltimateTicTacToeBoard::generate_zobrist_random(&mut seeded);
        let x_to_move_constant = seeded.gen::<u64>();

        ZobristConstants {
            x_constants: x_consts,
            o_constants: o_consts,
            x_to_move: x_to_move_constant,
        }
    }

    fn generate_zobrist_random(seeded: &mut rngs::StdRng) -> [u64; 81] {
        let x_consts = {
            let mut empty = [0u64; 81];
            for element in empty.iter_mut() {
                *element = seeded.gen::<u64>();
            }
            empty
        };
        x_consts
    }

    pub fn get_legal_moves_fast(&self) -> Vec<GameMove> {
        let complete_bbs = self.x_supers | self.o_supers | self.get_tied_mask();
        let mut moves = Vec::with_capacity(32);
        if (self.last_super_index >= 9) || ((complete_bbs & (1 << self.last_super_index)) != 0) {
            for subboard in 0..9usize {
                if (complete_bbs & (1 << subboard)) == 0 {
                    let occupied = self.xbb[subboard] | self.obb[subboard];
                    let base = (subboard as u8) * 9;
                    for sq in 0..9u8 {
                        if (occupied & (1 << sq)) == 0 {
                            moves.push(GameMove {
                                x_to_move: self.x_to_move,
                                move_index: base + sq,
                            });
                        }
                    }
                }
            }
        } else {
            let subboard = self.last_super_index as usize;
            let occupied = self.xbb[subboard] | self.obb[subboard];
            let base = self.last_super_index * 9;
            for sq in 0..9u8 {
                if (occupied & (1 << sq)) == 0 {
                    moves.push(GameMove {
                        x_to_move: self.x_to_move,
                        move_index: base + sq,
                    });
                }
            }
        }
        moves
    }
}

impl AlphaBetaCompatible<u64, UltimateTicTacToeUnmove, ZobristConstants>
    for UltimateTicTacToeBoard
{
    fn hash_val(&self) -> u64 {
        //TODO: Try to enforce a consistent hash value. As of now, hash_state is only correct
        //  for the case where to positions were make_move'd from the start position.
        self.hash_state
    }

    fn get_win_state(&self) -> GameResult {
        // check for X or O win patterns, check for a filled board, or return unresolved
        if RESULT_MASKS.iter().any(|x| (x & self.x_supers).eq(x)) {
            GameResult::P1Win
        } else if RESULT_MASKS.iter().any(|x| (x & self.o_supers).eq(x)) {
            GameResult::P2Win
        } else {
            let new_mask = self
                .xbb
                .iter()
                .zip(self.obb.iter())
                .enumerate()
                .filter_map(|(index, (xbb, obb))| match xbb | obb {
                    CAT => Some(1u16 << index),
                    _ => None,
                })
                .fold(0u16, |current, next| current | next);
            if new_mask | self.o_supers | self.x_supers == CAT {
                let x_won = self.x_supers.count_ones() as i8;
                let o_won = self.o_supers.count_ones() as i8;
                let heuristic_score = x_won - o_won;
                match heuristic_score {
                    0 => GameResult::Tie,
                    value if value > 0 => GameResult::P1Win,
                    _ => GameResult::P2Win,
                }
            } else {
                GameResult::Unresolved
            }
        }
    }

    fn x_to_move(&self) -> bool {
        self.x_to_move
    }

    fn make_move(
        &mut self,
        a_move: GameMove,
        zobrist_constants: &ZobristConstants,
    ) -> UltimateTicTacToeUnmove {
        let GameMove {
            x_to_move,
            move_index: a_move_index,
        } = a_move;

        let original_hash_state = self.hash_state;

        let hash_update: u64 = if x_to_move {
            zobrist_constants.x_to_move ^ zobrist_constants.x_constants[a_move_index as usize]
        } else {
            zobrist_constants.x_to_move ^ zobrist_constants.o_constants[a_move_index as usize]
        };
        self.hash_state ^= hash_update;

        let SubBoardAndSquare {
            sub_board_index,
            sub_square_index,
        } = parse_index(a_move_index);
        //            eprintln!("{} {}", sub_board_index, sub_square_index);

        let to_move_reset = x_to_move;
        let move_to_unmask = a_move_index;
        let mut a_super_unmask = 0;
        let sub_board_reset = self.last_super_index;

        match a_move {
            GameMove {
                x_to_move: true, ..
            } => {
                self.xbb[sub_board_index] ^= 1 << sub_square_index;
                let result_board = self.xbb[sub_board_index];

                let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                if is_sub_win {
                    let super_mask = 1u16 << sub_board_index;
                    self.x_supers |= super_mask;
                    a_super_unmask = super_mask;
                }
                self.last_super_index = sub_square_index as u8;
                self.x_to_move = false;
            }
            GameMove {
                x_to_move: false, ..
            } => {
                self.obb[sub_board_index] ^= 1 << sub_square_index;
                let result_board = self.obb[sub_board_index];

                let is_sub_win = RESULT_MASKS.iter().any(|x| (x & result_board).eq(x));

                if is_sub_win {
                    let super_mask = 1u16 << sub_board_index;
                    self.o_supers |= 1 << sub_board_index;
                    a_super_unmask = super_mask;
                }
                self.last_super_index = sub_square_index as u8;
                self.x_to_move = true;
            }
        }

        UltimateTicTacToeUnmove {
            x_to_move: to_move_reset,
            move_index: move_to_unmask,
            super_unmask: a_super_unmask,
            last_super: sub_board_reset,
            hash_reset: original_hash_state,
        }
    }

    fn unmake_move(&mut self, unmove: UltimateTicTacToeUnmove) {
        let UltimateTicTacToeUnmove {
            x_to_move,
            move_index,
            super_unmask,
            last_super,
            hash_reset,
        } = unmove;

        let SubBoardAndSquare {
            sub_board_index,
            sub_square_index,
        } = parse_index(move_index);

        self.x_to_move = x_to_move;
        self.last_super_index = last_super;
        self.hash_state = hash_reset;

        if x_to_move {
            self.xbb[sub_board_index] ^= 1 << sub_square_index;
            self.x_supers ^= super_unmask;
        } else {
            self.obb[sub_board_index] ^= 1 << sub_square_index;
            self.o_supers ^= super_unmask;
        };
    }

    fn score(&self) -> i8 {
        let x_won = self.x_supers.count_ones() as i8;
        let o_won = self.o_supers.count_ones() as i8;

        match self.get_win_state() {
            GameResult::P1Win => i8::MAX,
            GameResult::P2Win => i8::MIN,
            GameResult::Tie => 0i8,
            GameResult::Unresolved => x_won - o_won,
        }
    }

    fn get_legal_moves(&self, legality_lookup: &[Vec<GameMove>]) -> Vec<GameMove> {
        // TODO: P2: Modify this code to generate legal moves by an array lookup
        // TODO: P1: Consider generating those legal moves ordered by 'next best (small) TTT move'
        // gather won bitboards, the x/o occupation bitboards, and the total occupation
        // of the bitboard directed by the last move
        let complete_bbs = self.x_supers | self.o_supers | self.get_tied_mask();

        //            eprintln!("generating moves for x? {}",self.x_to_move);
        // arm 1 9 indicates 'play anywhere', as does directing to a full
        // arm 2 checks to see if the directed subboard has been won
        // arm 3 checks if the directed subboard is full
        // arm 4 creates a bitmask of 1s aligned to the directed board
        if (self.last_super_index >= 9) | ((complete_bbs & (1 << self.last_super_index)) != 0) {
            let mut moves = Vec::with_capacity(81);
            for subboard_index in 0..9usize {
                if (complete_bbs & (1 << subboard_index)) == 0 {
                    moves.append(&mut self.get_subboard_moves(
                        subboard_index,
                        legality_lookup,
                        self.x_to_move,
                    ))
                }
            }
            moves
        } else {
            self.get_subboard_moves(
                self.last_super_index as usize,
                legality_lookup,
                self.x_to_move,
            )
        }
    }

    fn display(&self) -> String {
        let mut rows = [['+'; 11]; 11];
        for row_index in 0..9 {
            let the_row = rows.get_mut(row_index + row_index / 3).unwrap();
            for col_index in 0..9 {
                let subboard_row = row_index % 3;
                let subboard_col = col_index % 3;

                let superboard_row = row_index / 3;
                let superboard_col = col_index / 3;

                let subboard = 3 * superboard_row + superboard_col;
                let subboard_square_mask = 1u16 << (3 * subboard_row + subboard_col);

                let an_item = the_row.get_mut(col_index + col_index / 3).unwrap();
                if self.xbb[subboard] & subboard_square_mask != 0 {
                    *an_item = 'X';
                } else if self.obb[subboard] & subboard_square_mask != 0 {
                    *an_item = 'O';
                } else {
                    *an_item = '.';
                }
            }
        }

        rows.iter()
            .map(|a_row| {
                let mut row = String::from_iter(a_row.iter());
                row.push('\n');
                row
            })
            .collect()
    }

    fn is_valid(&self) -> bool {
        self.obb
            .iter()
            .zip(self.xbb.iter())
            .all(|(o_subboard, x_subboard)| (o_subboard & x_subboard) == 0)
    }
}
