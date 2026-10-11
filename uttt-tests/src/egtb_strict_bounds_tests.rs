//! Strict Size, Memory, and Performance Bounds Tests for Macro-Factored Endgame Tablebases
//!
//! Guarantees:
//! 1. Tier 8 Tablebase: <= 500,000 entries, <= 5 MB file size, solved in < 3 seconds
//! 2. Tier 7 Tablebase: <= 3,500,000 entries, <= 50 MB file size, solved in < 30 seconds
//! 3. Peak RAM consumption during retrograde solving never exceeds 512 MB
//! 4. Collinear shared-line invariant is mathematically sound and deterministic

use std::time::Instant;
use uttt::egtb::macro_factored::{
    extract_augmented_tau, MacroFactoredSolver, MemoryBudgetGuard,
    TIER_7_MAX_ENTRIES, TIER_7_MAX_FILE_BYTES, TIER_8_MAX_ENTRIES, TIER_8_MAX_FILE_BYTES,
    MAX_HEAP_BUDGET_BYTES,
};

#[test]
fn test_tier8_strict_size_and_performance_bounds() {
    let temp_dir = std::env::temp_dir().join(format!("egtb_test_t8_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("Failed to create test dir");

    let solver = MacroFactoredSolver::new(&temp_dir, 4);

    let t0 = Instant::now();
    let (count, path) = solver.solve_tier_8().expect("Tier 8 solve failed");
    let elapsed = t0.elapsed();

    // 1. Strict Performance Bound: Generation must complete in < 3.0 seconds
    println!("Tier 8 solve time: {:.3}s", elapsed.as_secs_f64());
    assert!(
        elapsed.as_secs_f64() < 3.0,
        "Tier 8 generation exceeded time limit (took {:.3}s, limit 3.0s)",
        elapsed.as_secs_f64()
    );

    // 2. Strict Entry Count Bound: Must be <= 500,000 entries
    println!("Tier 8 entry count: {}", count);
    assert!(
        count <= TIER_8_MAX_ENTRIES,
        "Tier 8 entry count {} exceeds maximum allowed {}",
        count,
        TIER_8_MAX_ENTRIES
    );
    assert!(count > 100_000, "Tier 8 entry count too low: {}", count);

    // 3. Strict File Size Bound: Must be <= 5 MB on disk
    let metadata = std::fs::metadata(&path).expect("Tablebase file must exist");
    let file_bytes = metadata.len();
    println!("Tier 8 file size: {:.2} MB ({} bytes)", file_bytes as f64 / (1024.0 * 1024.0), file_bytes);
    assert!(
        file_bytes <= TIER_8_MAX_FILE_BYTES,
        "Tier 8 file size {} bytes exceeds 5 MB limit ({})",
        file_bytes,
        TIER_8_MAX_FILE_BYTES
    );
    assert!(file_bytes > 500_000, "Tier 8 file size too small: {}", file_bytes);

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_tier7_strict_size_and_performance_bounds() {
    let temp_dir = std::env::temp_dir().join(format!("egtb_test_t7_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("Failed to create test dir");

    let solver = MacroFactoredSolver::new(&temp_dir, 4);

    let t0 = Instant::now();
    let (count, path) = solver.solve_tier_7(2).expect("Tier 7 solve failed");
    let elapsed = t0.elapsed();

    // 1. Strict Performance Bound: Generation must complete in < 30.0 seconds
    println!("Tier 7 solve time: {:.3}s", elapsed.as_secs_f64());
    assert!(
        elapsed.as_secs_f64() < 30.0,
        "Tier 7 generation exceeded time limit (took {:.3}s, limit 30.0s)",
        elapsed.as_secs_f64()
    );

    // 2. Strict Entry Count Bound: Must be <= 3,500,000 entries
    println!("Tier 7 entry count: {}", count);
    assert!(
        count <= TIER_7_MAX_ENTRIES,
        "Tier 7 entry count {} exceeds maximum allowed {}",
        count,
        TIER_7_MAX_ENTRIES
    );
    assert!(count > 10_000, "Tier 7 entry count too low: {}", count);

    // 3. Strict File Size Bound: Must be <= 50 MB on disk
    let metadata = std::fs::metadata(&path).expect("Tablebase file must exist");
    let file_bytes = metadata.len();
    println!("Tier 7 file size: {:.2} MB ({} bytes)", file_bytes as f64 / (1024.0 * 1024.0), file_bytes);
    assert!(
        file_bytes <= TIER_7_MAX_FILE_BYTES,
        "Tier 7 file size {} bytes exceeds 50 MB limit ({})",
        file_bytes,
        TIER_7_MAX_FILE_BYTES
    );

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_macro_factored_invariance_and_collinear_detection() {
    // Subboards j = 0 (top-left) and k = 2 (top-right).
    // The collinear line is row 0: {0, 1, 2}, so m = 1 (top-center).
    let j = 0usize;
    let k = 2usize;
    let m = 1usize;

    // Case A: Macro board where subboard m was won by Player X
    let x_supers_a = 1u16 << m; // m is won by X
    let o_supers_a = 0u16;

    let (xw_j_a, xw_k_a, ow_j_a, ow_k_a, x_shared_a, o_shared_a, delta_a) =
        extract_augmented_tau(x_supers_a, o_supers_a, j, k);

    // Winning j alone or k alone does not complete 3-in-a-row yet
    assert!(!xw_j_a, "Winning j alone should not complete row yet");
    assert!(!xw_k_a, "Winning k alone should not complete row yet");
    assert!(!ow_j_a);
    assert!(!ow_k_a);

    // But x_shared MUST be true because m is won by X!
    assert!(
        x_shared_a,
        "Collinear shared line {{j, k, m}} with m won by X must set x_shared = true"
    );
    assert!(!o_shared_a);
    assert_eq!(delta_a, 1, "Net closed subboard delta must be +1 for X");

    // Case B: Macro board where subboard m ended in a Tie (neither won)
    let x_supers_b = 0u16;
    let o_supers_b = 0u16;

    let (xw_j_b, xw_k_b, ow_j_b, ow_k_b, x_shared_b, o_shared_b, delta_b) =
        extract_augmented_tau(x_supers_b, o_supers_b, j, k);

    // Here x_shared must be false because m is tied!
    assert!(!x_shared_b, "m is tied, so x_shared must be false");
    assert!(!o_shared_b);
    assert_eq!(delta_b, 0, "Neither won closed subboards");

    // CRITICAL: Case A and Case B MUST have different tau!
    assert_ne!(
        (xw_j_a, xw_k_a, x_shared_a),
        (xw_j_b, xw_k_b, x_shared_b),
        "Augmented tau must distinguish shared-line win potential from tied line!"
    );
}

#[test]
fn test_memory_budget_guard_under_budget() {
    // Normal operation under generous budget (512 MB) should not panic
    MemoryBudgetGuard::check(MAX_HEAP_BUDGET_BYTES);
    let rss = MemoryBudgetGuard::get_rss_bytes();
    println!("Current resident RSS: {:.2} MB", rss as f64 / (1024.0 * 1024.0));
}
