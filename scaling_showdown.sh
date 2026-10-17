#!/usr/bin/env bash
set -uo pipefail

LOG_FILE="scaling_showdown.log"
echo "=== Ultimate Tic-Tac-Toe Multi-Time-Control Scaling Showdown ===" | tee "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"

ARENA="./target/release/arena"
if [[ ! -f "$ARENA" ]]; then
    echo "Building arena in release mode..." | tee -a "$LOG_FILE"
    cargo build --release --bin arena
fi

TIME_CONTROLS=(30 100 200)

echo "--- Phase 1: AlphaBetaV2 vs Baseline AlphaBeta Scaling ---" | tee -a "$LOG_FILE"
for tc in "${TIME_CONTROLS[@]}"; do
    echo ">>> Running AlphaBetaV2 vs AlphaBeta at ${tc}ms (30 paired games)..." | tee -a "$LOG_FILE"
    $ARENA benchmark --bot alphabeta_v2 --baseline alphabeta --openings --opening-depth 2 --time-ms "$tc" --target-rd 40 --max-games 30 2>&1 | tee -a "$LOG_FILE"
    echo "Finished AlphaBetaV2 at ${tc}ms." | tee -a "$LOG_FILE"
done

echo "--- Phase 2: UmctsV3 vs Baseline UmctsV2 Scaling ---" | tee -a "$LOG_FILE"
for tc in "${TIME_CONTROLS[@]}"; do
    echo ">>> Running UmctsV3 vs UmctsV2 at ${tc}ms (30 paired games)..." | tee -a "$LOG_FILE"
    $ARENA benchmark --bot umcts_v3 --baseline umcts_v2 --openings --opening-depth 2 --time-ms "$tc" --target-rd 40 --max-games 30 2>&1 | tee -a "$LOG_FILE"
    echo "Finished UmctsV3 at ${tc}ms." | tee -a "$LOG_FILE"
done

echo "--- Phase 3: AlphaBetaV2 vs UmctsV2 Cross-Paradigm Scaling ---" | tee -a "$LOG_FILE"
for tc in "${TIME_CONTROLS[@]}"; do
    echo ">>> Running AlphaBetaV2 vs UmctsV2 at ${tc}ms (30 paired games)..." | tee -a "$LOG_FILE"
    $ARENA benchmark --bot alphabeta_v2 --baseline umcts_v2 --openings --opening-depth 2 --time-ms "$tc" --target-rd 40 --max-games 30 2>&1 | tee -a "$LOG_FILE"
    echo "Finished Cross-Paradigm at ${tc}ms." | tee -a "$LOG_FILE"
done

echo "=== Scaling Showdown Completed at: $(date) ===" | tee -a "$LOG_FILE"
