import os
import shutil
import numpy as np

def heal_checkpoint(checkpoint_path: str = "ml/replay_buffer_checkpoint.npz"):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found.")
        return False

    backup_path = checkpoint_path + ".bak"
    shutil.copyfile(checkpoint_path, backup_path)
    print(f"Backed up {checkpoint_path} -> {backup_path}")

    data = np.load(checkpoint_path)
    old_obs = data["observations"]
    old_mask = data["legal_masks"]
    old_pol = data["policies"]
    old_val = data["values"]
    old_meta = data["meta"]

    # Target partition boundaries
    new_gm_cap = 100_000
    new_deep_cap = 450_000
    new_sp_cap = 450_000
    new_cap = 1_000_000

    new_obs = np.zeros_like(old_obs)
    new_mask = np.zeros_like(old_mask)
    new_pol = np.zeros_like(old_pol)
    new_val = np.zeros_like(old_val)

    # 1. Migrate top 100k GM states (from [0 : 100,000])
    new_obs[:new_gm_cap] = old_obs[:new_gm_cap]
    new_mask[:new_gm_cap] = old_mask[:new_gm_cap]
    new_pol[:new_gm_cap] = old_pol[:new_gm_cap]
    new_val[:new_gm_cap] = old_val[:new_gm_cap]
    gm_count = new_gm_cap
    gm_ptr = 0

    # 2. Partition the 350,000 states from old deep_draw slice [300,000 : 650,000]
    dd_obs = old_obs[300_000:650_000]
    dd_mask = old_mask[300_000:650_000]
    dd_pol = old_pol[300_000:650_000]
    dd_val = old_val[300_000:650_000]

    is_draw = np.abs(dd_val.squeeze(1)) < 0.05
    is_decisive = ~is_draw

    draw_indices = np.where(is_draw)[0]
    decisive_indices = np.where(is_decisive)[0]

    n_draw = len(draw_indices)
    n_decisive = len(decisive_indices)
    print(f"Extracted from old deep_draw partition: {n_draw:,} draw states, {n_decisive:,} decisive self-play states.")

    # Populate new deep_draw partition [100,000 : 100,000 + n_draw]
    deep_start = new_gm_cap
    new_obs[deep_start : deep_start + n_draw] = dd_obs[draw_indices]
    new_mask[deep_start : deep_start + n_draw] = dd_mask[draw_indices]
    new_pol[deep_start : deep_start + n_draw] = dd_pol[draw_indices]
    new_val[deep_start : deep_start + n_draw] = dd_val[draw_indices]
    deep_count = n_draw
    deep_ptr = n_draw % new_deep_cap

    # Populate new self_play partition [550,000 : 550,000 + n_decisive]
    sp_start = new_gm_cap + new_deep_cap
    new_obs[sp_start : sp_start + n_decisive] = dd_obs[decisive_indices]
    new_mask[sp_start : sp_start + n_decisive] = dd_mask[decisive_indices]
    new_pol[sp_start : sp_start + n_decisive] = dd_pol[decisive_indices]
    new_val[sp_start : sp_start + n_decisive] = dd_val[decisive_indices]
    sp_count = n_decisive
    sp_ptr = n_decisive % new_sp_cap

    # 3. Old frozen self-play partition [650,000 : 1,000,000] is completely discarded!

    total_inserted = int(old_meta[10])
    new_meta = np.array([
        new_cap,
        new_gm_cap,
        new_deep_cap,
        new_sp_cap,
        gm_ptr,
        gm_count,
        deep_ptr,
        deep_count,
        sp_ptr,
        sp_count,
        total_inserted,
    ], dtype=np.int64)

    temp_path = checkpoint_path + ".tmp.npz"
    np.savez_compressed(
        temp_path,
        observations=new_obs,
        legal_masks=new_mask,
        policies=new_pol,
        values=new_val,
        meta=new_meta,
    )
    os.replace(temp_path, checkpoint_path)
    print(f"Healed checkpoint successfully saved to {checkpoint_path}!")
    print(f"Summary: GM: {gm_count:,}/{new_gm_cap:,} | Deep: {deep_count:,}/{new_deep_cap:,} | SP: {sp_count:,}/{new_sp_cap:,}")
    return True

if __name__ == "__main__":
    heal_checkpoint()
