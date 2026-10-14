#!/usr/bin/env python3
"""
Distributed Tier 6 Endgame Tablebase Worker for Google Colab and Cloud VMs.

Usage:
  python colab/tier6_worker.py --orbit 6 --threads 16 --out-dir /content/drive/MyDrive/uttt_egtb/tier6
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

CANONICAL_ORBITS = {
    0:  ("(0, 1, 2)", False, 107),
    1:  ("(0, 1, 3)", False, 263),
    2:  ("(0, 1, 5)", False, 186),
    3:  ("(0, 1, 6)", False, 271),
    4:  ("(0, 1, 7)", False, 207),
    5:  ("(0, 1, 8)", False, 255),
    6:  ("(0, 2, 6)", False, 389),
    7:  ("(0, 2, 7)", False, 203),
    8:  ("(0, 5, 7)", False, 99),
    9:  ("(1, 3, 5)", False, 159),
    10: ("(0, 1, 4)", True,  343),
    11: ("(0, 2, 4)", True,  397),
    12: ("(0, 4, 5)", True,  262),
    13: ("(0, 4, 8)", True,  99),
    14: ("(1, 3, 4)", True,  266),
    15: ("(1, 4, 7)", True,  78),
}

def run_cmd(cmd, env=None, check=True):
    print(f"\n[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True, env=env)
    if check and res.returncode != 0:
        print(f"[ERROR] Command failed with exit code {res.returncode}: {cmd}")
        sys.exit(res.returncode)
    return res.returncode

def ensure_rust():
    cargo_path = shutil.which("cargo")
    cargo_home_bin = os.path.expanduser("~/.cargo/bin/cargo")
    if cargo_path or os.path.exists(cargo_home_bin):
        os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
        print("[SETUP] Rust cargo already installed.")
        return

    print("[SETUP] Rust not found. Installing via rustup...")
    run_cmd("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"

def ensure_binary(repo_root):
    bin_path = os.path.join(repo_root, "target", "release", "solve_egtb")
    if not os.path.exists(bin_path):
        print("[BUILD] Compiling solve_egtb in release mode...")
        run_cmd(f"cd '{repo_root}' && cargo build --release --bin solve_egtb")
    else:
        print(f"[BUILD] Found existing solver binary: {bin_path}")
    return bin_path

def prepare_tier7(tier7_input, local_target="/content/tier7"):
    """
    Ensures Tier 7 tablebase is extracted to local fast SSD.
    """
    if os.path.isdir(tier7_input):
        print(f"[TIER7] Found existing Tier 7 directory at: {tier7_input}")
        return tier7_input

    if os.path.isfile(tier7_input):
        print(f"[TIER7] Extracting {tier7_input} to {local_target}...")
        os.makedirs(local_target, exist_ok=True)
        # Check zstd
        if not shutil.which("zstd"):
            run_cmd("sudo apt-get update -qq && sudo apt-get install -y -qq zstd")
        
        # Determine extraction flag
        extract_dir = os.path.dirname(local_target)
        run_cmd(f"tar -I zstd -xf '{tier7_input}' -C '{extract_dir}'")
        if os.path.isdir(local_target):
            print(f"[TIER7] Extraction complete -> {local_target}")
            return local_target
        # Check if extracted as 'tier7' in target
        candidate = os.path.join(extract_dir, "tier7")
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(f"Tier 7 tablebase not found at {tier7_input}. Please upload tier7_egtb.tar.zst.")

def main():
    parser = argparse.ArgumentParser(description="Distributed UTTT Tier 6 Tablebase Solver")
    parser.add_argument("--orbit", type=int, required=True, help="Canonical orbit index (0-15)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Worker threads")
    parser.add_argument("--tier7", type=str, default="/content/tier7", help="Tier 7 directory or tar.zst path")
    parser.add_argument("--out-dir", type=str, default="/content/drive/MyDrive/uttt_egtb/tier6", help="Output directory")
    parser.add_argument("--max-empty", type=int, default=4, help="Max empty cells (default: 4)")
    args = parser.parse_args()

    if args.orbit not in CANONICAL_ORBITS:
        print(f"[ERROR] Invalid orbit {args.orbit}. Valid orbits: 0-15")
        sys.exit(1)

    triplet, center_open, expected_taus = CANONICAL_ORBITS[args.orbit]
    print("=" * 65)
    print("  UTTT Distributed Tier 6 Tablebase Solver")
    print(f"  Target Orbit:        {args.orbit} | Triplet: {triplet}")
    print(f"  Center Open:         {center_open}")
    print(f"  Expected Tau Slices: {expected_taus}")
    print(f"  Worker Threads:      {args.threads}")
    print(f"  Output Directory:    {args.out_dir}")
    print("=" * 65)

    os.makedirs(args.out_dir, exist_ok=True)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    ensure_rust()
    bin_path = ensure_binary(repo_root)
    tier7_dir = prepare_tier7(args.tier7)

    # Check already completed slices
    pattern = os.path.join(args.out_dir, f"tier6_orbit_{args.orbit}_tau_*.bin")
    existing = len(glob.glob(pattern))
    print(f"[STATUS] Currently completed slices in out-dir: {existing}/{expected_taus}")

    if existing >= expected_taus:
        print(f"\n[COMPLETE] Orbit {args.orbit} is already 100% completed ({existing}/{expected_taus} slices)!")
        return

    # Launch solver
    cmd = (
        f"'{bin_path}' "
        f"--tier 6-zfs "
        f"--orbit {args.orbit} "
        f"--threads {args.threads} "
        f"--tier7-dir '{tier7_dir}' "
        f"--out-dir '{args.out_dir}' "
        f"--max-empty {args.max_empty}"
    )

    t0 = time.time()
    run_cmd(cmd)
    elapsed = time.time() - t0

    # Final check
    final_count = len(glob.glob(pattern))
    print(f"\n[FINISH] Solved Orbit {args.orbit} in {elapsed/3600.0:.2f} hours.")
    print(f"[FINISH] Slices present: {final_count}/{expected_taus}")

if __name__ == "__main__":
    main()
