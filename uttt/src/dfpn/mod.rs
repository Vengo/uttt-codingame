use std::collections::HashMap;
use crate::game_solving::{AlphaBetaCompatible, GameMove, GameResult};
use crate::packed_state::PackedBoardState;
use crate::uttt_trait_solving::{UltimateTicTacToeBoard, ZobristConstants};

pub const INFTY: u32 = 1_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProofNumbers {
    pub pn: u32,
    pub dn: u32,
}

impl ProofNumbers {
    #[inline(always)]
    pub fn new(pn: u32, dn: u32) -> Self {
        ProofNumbers { pn, dn }
    }

    #[inline(always)]
    pub fn is_proven(&self) -> bool {
        self.pn == 0
    }

    #[inline(always)]
    pub fn is_disproven(&self) -> bool {
        self.dn == 0
    }
}

pub struct DfPnSolver {
    pub tt: HashMap<PackedBoardState, ProofNumbers>,
    pub max_nodes: usize,
    pub nodes_visited: usize,
}

impl Default for DfPnSolver {
    fn default() -> Self {
        Self::new(500_000)
    }
}

impl DfPnSolver {
    pub fn new(max_nodes: usize) -> Self {
        DfPnSolver {
            tt: HashMap::with_capacity(65536),
            max_nodes,
            nodes_visited: 0,
        }
    }

    pub fn reset(&mut self) {
        self.tt.clear();
        self.nodes_visited = 0;
    }

    /// Solves whether the current player to move can force a win.
    /// Returns:
    /// - Some(true): Proven Win for player to move
    /// - Some(false): Disproven Win (Opponent wins or forced draw)
    /// - None: Search budget exceeded (unsolved)
    pub fn solve(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Option<bool> {
        self.nodes_visited = 0;
        let mut path = Vec::with_capacity(64);
        let root_pn = self.mid(board, INFTY, INFTY, &mut path, legal_moves_lookup, zobrist_constants);
        if root_pn.is_proven() {
            Some(true)
        } else if root_pn.is_disproven() {
            Some(false)
        } else {
            None
        }
    }

    /// Solves whether the current player to move can force a win, returning the best move if proven.
    /// Returns:
    /// - Some((true, Some(winning_move))): Proven Win for player to move, with winning move
    /// - Some((false, None)): Disproven Win (Opponent wins or forced draw)
    /// - None: Search budget exceeded (unsolved)
    pub fn solve_move(
        &mut self,
        board: &UltimateTicTacToeBoard,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> Option<(bool, Option<GameMove>)> {
        self.nodes_visited = 0;
        let mut path = Vec::with_capacity(64);
        let root_pn = self.mid(board, INFTY, INFTY, &mut path, legal_moves_lookup, zobrist_constants);
        if root_pn.is_proven() {
            let legal_moves = board.get_legal_moves(legal_moves_lookup);
            let attacker_win_result = if board.x_to_move { GameResult::P1Win } else { GameResult::P2Win };
            let mut winning_move = None;

            for m in legal_moves {
                let mut child = board.clone();
                child.make_move(m, zobrist_constants);
                if child.get_win_state() == attacker_win_result {
                    winning_move = Some(m);
                    break;
                }
                let child_key = child.to_packed_canonical();
                if let Some(&cached) = self.tt.get(&child_key) {
                    if cached.dn == 0 {
                        winning_move = Some(m);
                        break;
                    }
                }
            }

            Some((true, winning_move))
        } else if root_pn.is_disproven() {
            let legal_moves = board.get_legal_moves(legal_moves_lookup);
            let mut best_move = legal_moves.first().copied();
            let mut max_dn = 0;
            for m in legal_moves {
                let mut child = board.clone();
                child.make_move(m, zobrist_constants);
                let child_key = child.to_packed_canonical();
                if let Some(&cached) = self.tt.get(&child_key) {
                    if cached.dn > max_dn {
                        max_dn = cached.dn;
                        best_move = Some(m);
                    }
                }
            }
            Some((false, best_move))
        } else {
            None
        }
    }

    /// Depth-First Proof-Number search with thresholds (MID algorithm)
    /// Incorporates:
    /// 1. Canonical 167-bit PackedBoardState transposition table keys
    /// 2. 1-ply immediate attacker win pruning
    /// 3. 2-ply Grandfather refutation lookahead (defender immediate wins)
    /// 4. Path-cycle detection
    fn mid(
        &mut self,
        board: &UltimateTicTacToeBoard,
        th_pn: u32,
        th_dn: u32,
        path: &mut Vec<PackedBoardState>,
        legal_moves_lookup: &[Vec<GameMove>],
        zobrist_constants: &ZobristConstants,
    ) -> ProofNumbers {
        self.nodes_visited += 1;
        let key = board.to_packed_canonical();

        // 1. Check transposition table hit
        if let Some(&cached) = self.tt.get(&key) {
            if cached.pn >= th_pn || cached.dn >= th_dn || cached.is_proven() || cached.is_disproven() {
                return cached;
            }
        }

        // 2. Check terminal state
        let win_state = board.get_win_state();
        if win_state != GameResult::Unresolved {
            let res = match win_state {
                GameResult::P1Win => {
                    if board.x_to_move {
                        ProofNumbers::new(0, INFTY) // Attacker is P1 and won
                    } else {
                        ProofNumbers::new(INFTY, 0) // Opponent won
                    }
                }
                GameResult::P2Win => {
                    if !board.x_to_move {
                        ProofNumbers::new(0, INFTY) // Attacker is P2 and won
                    } else {
                        ProofNumbers::new(INFTY, 0) // Opponent won
                    }
                }
                _ => ProofNumbers::new(INFTY, 0), // Tie treated as not win
            };
            self.tt.insert(key, res);
            return res;
        }

        let legal_moves = board.get_legal_moves(legal_moves_lookup);
        if legal_moves.is_empty() {
            let res = ProofNumbers::new(INFTY, 0);
            self.tt.insert(key, res);
            return res;
        }

        // Budget check
        if self.nodes_visited >= self.max_nodes {
            return *self.tt.entry(key).or_insert(ProofNumbers::new(1, 1));
        }

        // OR-node semantics: current player to move is attacker
        // Generate children with 1-ply win and 2-ply Grandfather lookahead
        let mut children: Vec<(GameMove, UltimateTicTacToeBoard, ProofNumbers)> = Vec::with_capacity(legal_moves.len());
        let attacker_win_result = if board.x_to_move { GameResult::P1Win } else { GameResult::P2Win };
        let defender_win_result = if board.x_to_move { GameResult::P2Win } else { GameResult::P1Win };

        for m in legal_moves {
            let mut child = board.clone();
            child.make_move(m, zobrist_constants);
            let child_key = child.to_packed_canonical();

            // 1-ply lookahead: did attacker's move immediately win the game?
            if child.get_win_state() == attacker_win_result {
                let win_pn = ProofNumbers::new(0, INFTY);
                self.tt.insert(key, win_pn);
                self.tt.insert(child_key, ProofNumbers::new(INFTY, 0));
                return win_pn;
            }

            // Path cycle detection
            if path.contains(&child_key) {
                // Cycle back to ancestor: cannot prove attacker win through repetition
                children.push((m, child, ProofNumbers::new(INFTY, 0)));
                continue;
            }

            // Check TT for child
            if let Some(&cached_child) = self.tt.get(&child_key) {
                children.push((m, child, cached_child));
                continue;
            }

            // 2-ply Grandfather lookahead: can defender immediately win from child?
            let defender_moves = child.get_legal_moves(legal_moves_lookup);
            if defender_moves.is_empty() {
                // Defender has no moves and child is not attacker win -> draw (disproven)
                let disproven = ProofNumbers::new(INFTY, 0);
                self.tt.insert(child_key, disproven);
                children.push((m, child, disproven));
                continue;
            }

            let mut defender_wins_immediately = false;
            for &dm in &defender_moves {
                let mut grandchild = child.clone();
                grandchild.make_move(dm, zobrist_constants);
                if grandchild.get_win_state() == defender_win_result {
                    defender_wins_immediately = true;
                    break;
                }
            }

            if defender_wins_immediately {
                // Defender has an immediate counter-refutation win! Child is disproven as attacker win.
                let disproven = ProofNumbers::new(INFTY, 0);
                self.tt.insert(child_key, disproven);
                children.push((m, child, disproven));
            } else {
                children.push((m, child, ProofNumbers::new(1, 1)));
            }
        }

        // MID iterative threshold expansion
        path.push(key);

        loop {
            // Compute current node numbers
            let mut best_idx = 0;
            let mut min_pn = INFTY;
            let mut second_min_pn = INFTY;
            let mut sum_dn = 0u32;

            for (i, (_, _, c_pn)) in children.iter().enumerate() {
                // Child is AND-node from attacker perspective:
                // child's dn is attacker's effective pn, child's pn is attacker's effective dn
                let child_effective_pn = c_pn.dn;
                let child_effective_dn = c_pn.pn;

                if child_effective_pn < min_pn {
                    second_min_pn = min_pn;
                    min_pn = child_effective_pn;
                    best_idx = i;
                } else if child_effective_pn < second_min_pn {
                    second_min_pn = child_effective_pn;
                }

                sum_dn = sum_dn.saturating_add(child_effective_dn);
            }

            let cur_pn = min_pn;
            let cur_dn = sum_dn.min(INFTY);

            if cur_pn >= th_pn || cur_dn >= th_dn || cur_pn == 0 || cur_dn == 0 || self.nodes_visited >= self.max_nodes {
                let res = ProofNumbers::new(cur_pn, cur_dn);
                self.tt.insert(key, res);
                path.pop();
                return res;
            }

            // Calculate child thresholds
            let child_th_pn = th_dn.saturating_sub(cur_dn.saturating_sub(children[best_idx].2.pn)).min(INFTY);
            let child_th_dn = th_pn.min(second_min_pn.saturating_add(1));

            let child_board = children[best_idx].1.clone();
            let new_child_pn = self.mid(&child_board, child_th_pn, child_th_dn, path, legal_moves_lookup, zobrist_constants);
            children[best_idx].2 = new_child_pn;
        }
    }
}
