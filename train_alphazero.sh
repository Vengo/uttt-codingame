#!/usr/bin/env bash
set -uo pipefail

CYCLES=${1:-5}
GAMES=${2:-32}
EPOCHS=${3:-3}
LOG_FILE="alphazero_training.log"

echo "============================================================" | tee -a "$LOG_FILE"
echo "  AlphaZero Self-Play & Continual Improvement Daemon" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Configuration: Cycles=$CYCLES, Games/Cycle=$GAMES, Epochs=$EPOCHS" | tee -a "$LOG_FILE"

./ml/.venv/bin/python ml/train_cycle.py --iterations "$CYCLES" --games "$GAMES" --epochs "$EPOCHS" 2>&1 | tee -a "$LOG_FILE"

echo "Rebuilding release arena binary with updated neural weights..." | tee -a "$LOG_FILE"
cargo build --release --bin arena 2>&1 | tee -a "$LOG_FILE"

echo "Running 10-game benchmark match against RuleBasedBot..." | tee -a "$LOG_FILE"
./target/release/arena match --bot1 alphazero --bot2 rule_based --games 10 --time-ms 50 2>&1 | tee -a "$LOG_FILE"

echo "=== AlphaZero Improvement Cycle Completed at $(date) ===" | tee -a "$LOG_FILE"
