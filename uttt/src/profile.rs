use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::game_solving::{AlphaBetaCompatible, IdentityHasherBuilder, UmctsNode};
use crate::uttt_trait_solving::UltimateTicTacToeBoard;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CacheProfileResult {
    pub small_table_entries: usize,
    pub small_table_kb: usize,
    pub small_table_latency_ns: f64,
    pub large_table_entries: usize,
    pub large_table_kb: usize,
    pub large_table_latency_ns: f64,
    pub cache_slowdown_factor: f64,
    pub summary: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AllocatorProfileResult {
    pub total_allocations: usize,
    pub elapsed_ms: f64,
    pub allocations_per_sec: f64,
    pub ns_per_allocation: f64,
    pub summary: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchThroughputResult {
    pub bot_name: String,
    pub rollouts_per_sec: f64,
    pub average_depth: f64,
    pub summary: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HardwareProfileReport {
    pub cpu_info: String,
    pub cache_profile: CacheProfileResult,
    pub allocator_profile: AllocatorProfileResult,
    pub search_throughput: Vec<SearchThroughputResult>,
}

/// Profiles CPU cache hierarchy latency: L3 cache-resident table vs DRAM-spilling table.
pub fn profile_cpu_cache() -> CacheProfileResult {
    let small_entries = 50_000usize;
    let large_entries = 2_000_000usize;
    let lookups = 1_000_000usize;

    // Small table (50k * ~25B ~ 1.25 MB, comfortably resident in CPU L3 cache)
    let mut small_store: HashMap<u64, UmctsNode, IdentityHasherBuilder> =
        HashMap::with_capacity_and_hasher(small_entries, IdentityHasherBuilder {});
    for i in 0..small_entries as u64 {
        let k = i.wrapping_mul(11400714819323198485);
        small_store.insert(k, UmctsNode::default());
    }

    let t0 = Instant::now();
    let mut small_found = 0usize;
    for i in 0..lookups {
        let k = ((i % small_entries) as u64).wrapping_mul(11400714819323198485);
        if small_store.get(&k).is_some() {
            small_found += 1;
        }
    }
    let small_elapsed = t0.elapsed();
    let small_latency_ns = (small_elapsed.as_nanos() as f64) / (lookups as f64);
    assert_eq!(small_found, lookups);

    // Large table (2M entries ~ 50-60 MB, exceeds standard L3 cache and spills to DDR4 DRAM)
    let mut large_store: HashMap<u64, UmctsNode, IdentityHasherBuilder> =
        HashMap::with_capacity_and_hasher(large_entries, IdentityHasherBuilder {});
    for i in 0..large_entries as u64 {
        let k = i.wrapping_mul(11400714819323198485);
        large_store.insert(k, UmctsNode::default());
    }

    let t1 = Instant::now();
    let mut large_found = 0usize;
    for i in 0..lookups {
        let k = ((i % large_entries) as u64).wrapping_mul(11400714819323198485);
        if large_store.get(&k).is_some() {
            large_found += 1;
        }
    }
    let large_elapsed = t1.elapsed();
    let large_latency_ns = (large_elapsed.as_nanos() as f64) / (lookups as f64);
    assert_eq!(large_found, lookups);

    let slowdown = large_latency_ns / small_latency_ns.max(0.1);
    let summary = format!(
        "L3 Cache-Resident table: {:.1} ns/lookup vs DRAM Spilling table: {:.1} ns/lookup ({:.2}x slowdown factor)",
        small_latency_ns, large_latency_ns, slowdown
    );

    CacheProfileResult {
        small_table_entries: small_entries,
        small_table_kb: (small_entries * 25) / 1024,
        small_table_latency_ns: small_latency_ns,
        large_table_entries: large_entries,
        large_table_kb: (large_entries * 25) / 1024,
        large_table_latency_ns: large_latency_ns,
        cache_slowdown_factor: slowdown,
        summary,
    }
}

/// Profiles dynamic heap allocation throughput, reproducing small-buffer churn.
pub fn profile_allocator() -> AllocatorProfileResult {
    let iterations = 200_000usize;
    let t0 = Instant::now();

    for i in 0..iterations {
        let cap = (i % 32) + 1;
        let mut v: Vec<u8> = Vec::with_capacity(cap);
        v.push((i & 0xFF) as u8);
        std::hint::black_box(v);
    }

    let elapsed = t0.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let allocs_per_sec = (iterations as f64) / elapsed.as_secs_f64();
    let ns_per_alloc = (elapsed.as_nanos() as f64) / (iterations as f64);

    let summary = format!(
        "{:.2} million allocations/sec ({:.1} ns per heap alloc/free)",
        allocs_per_sec / 1_000_000.0,
        ns_per_alloc
    );

    AllocatorProfileResult {
        total_allocations: iterations,
        elapsed_ms,
        allocations_per_sec: allocs_per_sec,
        ns_per_allocation: ns_per_alloc,
        summary,
    }
}

/// Measures simulation rollout speed (pure random playouts to terminal)
pub fn profile_simulation_throughput() -> SearchThroughputResult {
    let board = UltimateTicTacToeBoard::default();
    let zobrist_constants = UltimateTicTacToeBoard::get_zobrist_constants();
    let mut rng = SmallRng::from_entropy();

    let duration = Duration::from_millis(500);
    let t0 = Instant::now();
    let mut rollouts = 0usize;
    let mut total_depth = 0usize;

    while t0.elapsed() < duration {
        let mut sim_board = board.clone();
        let mut depth = 0;
        while sim_board.get_win_state() == crate::game_solving::GameResult::Unresolved && depth < 60 {
            let moves = sim_board.get_legal_moves_fast();
            if moves.is_empty() {
                break;
            }
            let chosen = *moves.choose(&mut rng).unwrap();
            sim_board.make_move(chosen, &zobrist_constants);
            depth += 1;
        }
        total_depth += depth;
        rollouts += 1;
    }

    let actual_sec = t0.elapsed().as_secs_f64();
    let rollouts_per_sec = (rollouts as f64) / actual_sec;
    let average_depth = (total_depth as f64) / (rollouts.max(1) as f64);

    SearchThroughputResult {
        bot_name: "Pure Random Simulation (No Tree Alloc)".to_string(),
        rollouts_per_sec,
        average_depth,
        summary: format!(
            "{:.0} rollouts/sec (avg depth: {:.1} plies/rollout)",
            rollouts_per_sec, average_depth
        ),
    }
}

/// Runs full hardware and search profiling report
pub fn run_hardware_profile() -> HardwareProfileReport {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|info| {
            info.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let cache_profile = profile_cpu_cache();
    let allocator_profile = profile_allocator();
    let throughput = profile_simulation_throughput();

    HardwareProfileReport {
        cpu_info,
        cache_profile,
        allocator_profile,
        search_throughput: vec![throughput],
    }
}
