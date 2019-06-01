use std::io;
use std::collections::HashMap;

const TTT_BOARD_WIDTH: usize = 3;

// 256 128 64 32 16 8 4 2 1
// 1 2 4 8 16 32 64 128 256
const RESULT_MASKS: [u16; 8] = [7,7<<3,7<<6,73, 73<<1, 73<<2, 84, 273];
/// 0 1 2 O . X
/// 3 4 5 X X O
/// 6 7 8 O . .
const MOVE_ORDERING: [u8; 9] = [4,0,2,6,3,5,8,1,7];
//const MOVE_ORDERING: [u8; 9] = [1,3,5,7,8,4,6,2,0];
const CAT: u16 = (1<<9) -1;

struct TTTBoard{
    xbb: u16,
    obb: u16,
    x_to_move: bool,
}

struct TTTMove {
    x_to_move: bool,
    move_index: u8,
}


impl TTTBoard{
    fn alpha_beta_search(&mut self, depth: &u8, alpha: &i8, beta: &i8) -> (i8,u8) {
        ///from https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
        /// function alphabeta(node, depth, α, β, maximizingPlayer) is
        ///    if depth = 0 or node is a terminal node then
        ///        return the heuristic value of node
        ///    if maximizingPlayer then
        ///        value := −∞
        ///        for each child of node do
        ///            value := max(value, alphabeta(child, depth − 1, α, β, FALSE))
        ///            α := max(α, value)
        ///            if α ≥ β then
        ///                break (* β cut-off *)
        ///        return value
        ///    else
        ///        value := +∞
        ///        for each child of node do
        ///            value := min(value, alphabeta(child, depth − 1, α, β, TRUE))
        ///            β := min(β, value)
        ///            if α ≥ β then
        ///                break (* α cut-off *)
        ///        return value

        match (*depth, self.score()) {
            (a_depth, a_score ) if ((a_score == 127) & self.x_to_move) => {
                return (0,255);
            }
            _ => panic!("alpha_beta search isn't really implemented yet..."),
        }

        eprintln!("testing build");
        return (0,255);
    }

    fn hash_val(&self) -> u32 {
        let base_hash = (self.xbb as u32)<<16 | (self.obb as u32);
        return match self.x_to_move {
            true => { base_hash },
            false => { !base_hash },
        };
//        return 0;
    }

    fn min_max_hash(&mut self, mut transpositions: &mut HashMap<u32,i8>) -> (i8, u8) {
        match self.get_win_state() {
            TTTState::XWin => {
                let score = 127i8;
                let move_index = 255u8;
                transpositions.insert(self.hash_val(), score);
                return (score, move_index);
            },
            TTTState::OWin => {
                let score = -128i8;
                let move_index = 255u8;
                transpositions.insert(self.hash_val(), score);
                return (score, move_index);
            },
            TTTState::Cat => {
                let score = 0i8;
                let move_index = 255u8;
                transpositions.insert(self.hash_val(), score);
                return (score, move_index);
            },
            TTTState::Unresolved => {
                match self.x_to_move {
                    true => {
                        let mut max_score = -128;
                        let mut max_move_index = 0;

                        for each_move in self.get_legal_moves() {
                            self.make_move(&each_move);
                            let new_hash = self.hash_val();
                            let (move_score, _move_index) = match transpositions.get(&new_hash){
                                Some(hash_value) if *hash_value==127 => {
                                    self.unmake_move(&each_move);
                                    return (hash_value.clone(), each_move.move_index)
                                },
                                Some(hash_value) => {
                                    (hash_value.clone(), each_move.move_index)
                                },
                                None => {
                                    self.min_max_hash(&mut transpositions)
                                },
                            };
                            match move_score {
                                127 => {
                                    transpositions.insert(self.hash_val(),move_score);
                                    self.unmake_move(&each_move);
                                    return (move_score, each_move.move_index);
                                },
                                move_score if move_score > max_score => {
                                    max_score = move_score;
                                    max_move_index = each_move.move_index;
                                },
                                _move_score => {
                                    //pass
                                },
                            };
                            self.unmake_move(&each_move);
                        };
                        return (max_score, max_move_index);
                    },
                    false => {
                        let mut min_score = 127;
                        let mut min_move_index = 0;

                        for each_move in self.get_legal_moves() {
                            self.make_move(&each_move);
                            let new_hash = self.hash_val();
                            let (move_score, _move_index) = match transpositions.get(&new_hash){
                                Some(hash_value) if *hash_value==-128 => {
                                    self.unmake_move(&each_move);
                                    return (hash_value.clone(), each_move.move_index)
                                },
                                Some(hash_value) => {
                                    (hash_value.clone(), each_move.move_index)
                                },
                                None => {
                                    self.min_max_hash(&mut transpositions)
                                },
                            };
                            match move_score {
                                -128 => {
                                    transpositions.insert(self.hash_val(),move_score);
                                    self.unmake_move(&each_move);
                                    return (move_score, each_move.move_index);
                                },
                                move_score if move_score < min_score => {
                                    min_score = move_score;
                                    min_move_index = each_move.move_index;
                                },
                                _move_score => {
                                    //pass
                                },
                            };
                            self.unmake_move(&each_move);
                        };
                        return (min_score, min_move_index);
                    },
                }
            }
        };
    }

    fn is_valid(&self) -> bool {
        // there are more ways to invalidate a board than are checked here
        return (self.obb & self.xbb) == 0;
    }

    fn make_move(&mut self, a_move: &TTTMove) {
        match a_move {
            TTTMove{x_to_move: true, move_index} => {
                self.xbb ^= 1u16 << move_index;
                self.x_to_move = false;
            },
            TTTMove{x_to_move: false, move_index} => {
                self.obb ^= 1u16 << move_index;
                self.x_to_move = true;
            },
        }
    }

    fn unmake_move(&mut self, a_move: &TTTMove) {
        match a_move {
            TTTMove{x_to_move: true, move_index} => {
                self.xbb ^= 1u16 << move_index;
                self.x_to_move = true;
            },
            TTTMove{x_to_move: false, move_index} => {
                self.obb ^= 1u16 << move_index;
                self.x_to_move = false;
            },
        }

    }

    fn get_win_state(&self) -> TTTState {
        // check for X or O win patterns, check for a filled board, or return unresolved
        if RESULT_MASKS.iter().any(|x| (x & self.xbb).eq(x) ) {
            return TTTState::XWin;
        } else if RESULT_MASKS.iter().any(|x| (x & self.obb).eq(x) ) {
            return TTTState::OWin;
        } else if (self.xbb|self.obb)==CAT {
            return TTTState::Cat;
        } else {
            return TTTState::Unresolved;
        };
    }

    fn get_legal_moves(&self) -> Vec<TTTMove> {
        let occupied_bb = self.xbb | self.obb;

        let valid_moves = MOVE_ORDERING.iter().filter_map(|a_move_index|{
            match occupied_bb& (1u16<<a_move_index) {
                0 => Some(TTTMove{x_to_move: self.x_to_move, move_index: a_move_index.clone()}),
                _nonzero => {
                    None
                },
            }
        }).collect();

        return valid_moves;
    }

    fn score(&self) -> i8 {
        match self.get_win_state() {
            TTTState::XWin => { 127 },
            TTTState::OWin => {-128 },
            TTTState::Cat => { 0 },
            TTTState::Unresolved if self.x_to_move => { 1 },
            TTTState::Unresolved => { -1 },
        }
    }

    fn display(&self) -> String {
//        let mut board_rep = String::new();
        let mut rows = Vec::new();
        for row in 0..TTT_BOARD_WIDTH {
            let mut row_rep = String::new();
            for col in 0..TTT_BOARD_WIDTH {
                let mask = 1 << (col+ TTT_BOARD_WIDTH*row);
                if self.xbb & mask !=0{
                    row_rep.push('X');
                } else if self.obb & mask !=0{
                    row_rep.push('O');
                } else {
                    row_rep.push('.');
                }
            }
            rows.push(row_rep);
        }
        return rows.join("\n").to_string();
    }
}

#[derive(Debug, Eq, PartialEq)]
enum TTTState {
    XWin,
    OWin,
    Cat,
    Unresolved,
}


macro_rules! parse_input {
    ($x:expr, $t:ident) => ($x.trim().parse::<$t>().unwrap())
}

/**
 * Auto-generated code below aims at helping you parse
 * the standard input according to the problem statement.
 **/


fn main() {
    // game loop

    let mut ttt_board = TTTBoard{
        xbb: 0,
        obb: 0,
        x_to_move: true,
    };

    let mut transposition_table = HashMap::new();

    loop {
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let inputs = input_line.split(" ").collect::<Vec<_>>();
        let opponent_row = parse_input!(inputs[0], i32);
        let opponent_col = parse_input!(inputs[1], i32);

        eprintln!("opponent row was: {} {}", opponent_row, opponent_col);
        eprintln!("start of turn board state was:\n{}",ttt_board.display());
        match (opponent_row,opponent_col) {
            (-1,-1) => {
                let (best_value, best_action) = ttt_board.min_max_hash(&mut transposition_table);
                eprintln!("best value: {}\nbest action: {}",best_value,best_action);
                let my_move = TTTMove{move_index: best_action, x_to_move:ttt_board.x_to_move};
                ttt_board.make_move(&my_move);
                println!("{} {}",best_action/3, best_action%3);
            },
            (row,col) => {
                let opp_move_index = (3*row+col) as u8;

                let opp_move = TTTMove{move_index: opp_move_index, x_to_move: ttt_board.x_to_move};
                ttt_board.make_move(&opp_move);
                eprintln!("board after opponent move was:\n{}",ttt_board.display());

                let (best_value, best_action) = ttt_board.min_max_hash(&mut transposition_table);
                let my_move = TTTMove{move_index: best_action, x_to_move:ttt_board.x_to_move};
                ttt_board.make_move(&my_move);

                eprintln!("best value: {}\nbest action: {}",best_value,best_action);
                println!("{} {}",best_action/3, best_action%3);
            },
        }
        let mut input_line = String::new();
        io::stdin().read_line(&mut input_line).unwrap();
        let valid_action_count = parse_input!(input_line, i32);
        eprintln!("Valid moves:");

        let mut actions = Vec::new();

        for _i in 0..valid_action_count as usize {
            let mut input_line = String::new();
            io::stdin().read_line(&mut input_line).unwrap();
            let inputs = input_line.split(" ").collect::<Vec<_>>();
            let row = parse_input!(inputs[0], i32);
            let col = parse_input!(inputs[1], i32);
            actions.push((row,col));
            eprintln!("\t{} {}", row,col);
        }
//        println!("{} {}",actions[0].0, actions[0].1);
    }
}


#[cfg(test)]
mod test{
    use crate::{TTTBoard, TTTMove, TTTState, RESULT_MASKS};
    use std::collections::HashMap;

    #[test]
    fn testprint() {
        eprintln!("printing to stderr");
        println!("printing to stdout");
        let x = 2;
        assert_eq!(x,2);
    }

    #[test]
    fn display_test1() {
        let x = TTTBoard{
            xbb: 1<<8,
            obb: 0<<8,
            x_to_move: false,
        };
        let board_string = x.display();
        assert_eq!(board_string,"...\n...\n..X");
    }

    #[test]
    fn display_test2() {
        let x = TTTBoard{
            xbb: 1<<8,
            obb: 1<<2,
            x_to_move: false,
        };
        let board_string = x.display();
        assert_eq!(board_string,"..O\n...\n..X");
    }

    #[test]
    fn make_and_unmake_move_test_1() {
        let mut x = TTTBoard{
            xbb: 0,
            obb: 0,
            x_to_move: true,
        };
        let board_string = x.display();
        assert_eq!(board_string,"...\n...\n...");
        assert_eq!(x.x_to_move,true);


        //testing an x move
        let ttt_move = TTTMove{move_index: 0, x_to_move:true};
        x.make_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"X..\n...\n...");
        assert_eq!(x.x_to_move,false);

        // testing an x unmake move
        x.unmake_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"...\n...\n...");
        assert_eq!(x.x_to_move,true);

        //readding an x move
        let ttt_move = TTTMove{move_index: 4, x_to_move:true};
        x.make_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"...\n.X.\n...");
        assert_eq!(x.x_to_move,false);

        //testing an o move
        let ttt_move = TTTMove{move_index: 1, x_to_move:false};
        x.make_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,".O.\n.X.\n...");
        assert_eq!(x.x_to_move,true);

        //unmaking an o move
        x.unmake_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"...\n.X.\n...");
        assert_eq!(x.x_to_move,false);

        //testing an o move
        let ttt_move = TTTMove{move_index: 0, x_to_move:false};
        x.make_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"O..\n.X.\n...");
        assert_eq!(x.x_to_move,true);

        //testing an x move
        let ttt_move = TTTMove{move_index: 1, x_to_move:true};
        x.make_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"OX.\n.X.\n...");
        assert_eq!(x.x_to_move,false);

        //testing an o move
        let ttt_move = TTTMove{move_index: 2, x_to_move:false};
        x.make_move(&ttt_move);
        let board_string = x.display();
        assert_eq!(board_string,"OXO\n.X.\n...");
        assert_eq!(x.x_to_move,true);

        //testing an x move
        let ttt_move = TTTMove{move_index: 7, x_to_move:true};
        x.make_move(&ttt_move);
        let board_string = x.display();
        RESULT_MASKS.iter().for_each(| mask| println!{"mask:\t{}\nbitw:\t{}\nxbb:\t{}\n",mask, x.xbb&mask, x.xbb});

        assert_eq!(board_string,"OXO\n.X.\n.X.");
        assert_eq!(x.x_to_move,false);
        assert_eq!(x.get_win_state(), TTTState::XWin);
    }

    #[test]
    fn x_win_tests() {
        let test_board = TTTBoard{ xbb: 7, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"XXX\n...\n...");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 7<<3, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"...\nXXX\n...");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 7<<6, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"...\n...\nXXX");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 73, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"X..\nX..\nX..");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 73<<1, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),".X.\n.X.\n.X.");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 73<<2, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"..X\n..X\n..X");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 273, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"X..\n.X.\n..X");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

        let test_board = TTTBoard{ xbb: 84, obb: 0, x_to_move: true, };
        assert_eq!(test_board.display(),"..X\n.X.\nX..");
        assert_eq!(test_board.get_win_state(), TTTState::XWin);

    }

    #[test]
    fn o_win_tests() {
        let test_board = TTTBoard{ xbb: 0, obb: 7, x_to_move: true, };
        assert_eq!(test_board.display(),"OOO\n...\n...");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 7<<3, x_to_move: true, };
        assert_eq!(test_board.display(),"...\nOOO\n...");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 7<<6, x_to_move: true, };
        assert_eq!(test_board.display(),"...\n...\nOOO");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 73, x_to_move: true, };
        assert_eq!(test_board.display(),"O..\nO..\nO..");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 73<<1, x_to_move: true, };
        assert_eq!(test_board.display(),".O.\n.O.\n.O.");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 73<<2, x_to_move: true, };
        assert_eq!(test_board.display(),"..O\n..O\n..O");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 273, x_to_move: true, };
        assert_eq!(test_board.display(),"O..\n.O.\n..O");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

        let test_board = TTTBoard{ xbb: 0, obb: 84, x_to_move: true, };
        assert_eq!(test_board.display(),"..O\n.O.\nO..");
        assert_eq!(test_board.get_win_state(), TTTState::OWin);

    }

    #[test]
    fn movegen_test_1() {
        let mut x = TTTBoard{
            xbb: 0,
            obb: 0,
            x_to_move: true,
        };

        while x.get_legal_moves().len()>0 {
            let moves = x.get_legal_moves();
            x.make_move(&moves[0]);
            eprintln!("{}",x.display());
            assert_eq!(true, x.is_valid());
        }
    }

    #[test]
    fn min_max_hash_test() {
        let mut ttt_board = TTTBoard{
            xbb: 0,
            obb: 0,
            x_to_move: true,
        };

        let mut ttable = HashMap::new();
        let (board_val, move_index) = ttt_board.min_max_hash(&mut ttable);
        assert_eq!((0,4), (board_val,move_index));
//        eprintln!("{}",final_board);
        assert_eq!((0, 4),ttt_board.min_max_hash(&mut ttable));
        assert_eq!(ttable.len(),1217);

        while ttt_board.get_legal_moves().len() > 0 {
            let (board_val, move_index) = ttt_board.min_max_hash(&mut ttable);
            let a_move = TTTMove{ move_index, x_to_move: ttt_board.x_to_move};
            ttt_board.make_move(&a_move);
            assert_eq!(board_val,0);
        }
        assert_eq!(ttt_board.display(),"OOX\nXXO\nOXX");
    }
}