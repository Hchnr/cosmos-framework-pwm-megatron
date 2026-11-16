# Fixed UND/GEN training profile

The 4-GPU and 8-GPU fixed-static T2V recipes use **UND=1086, GEN=64450**, preserving the total length of **65536** and GA=2:

- [4-GPU smoke recipe](../examples/toml/sft_config/vision_sft_nano_t2v_4gpu_fixed_static.toml)
- [8-GPU main recipe](../examples/toml/sft_config/vision_sft_nano_t2v_8gpu_fixed_static.toml)

Dynamic recipes do not use a fixed UND/GEN split and remain unchanged. Dataset batch ratios remain 16:25; this is separate from the token budgets within each pack.

## Evidence and scope

In the 2026-09-08 study at 480p on 8×H100, the 1086/64450 profile improved effective GEN throughput by **4.2641%** over 5958/59578 in one matched run with 20 warmup steps and 50 measured steps. This was the highest throughput among completed profiles. No repeated-seed confidence interval, 4-GPU or 128-GPU throughput validation, or global-optimality claim is available.

The measured resolution was supplied through launcher overrides. Job names and VAE warmup settings alone do not set model/dataloader resolution. Use the paired launcher with `TRAIN_RESOLUTION=480` for that resolution; see [training documentation](training.md).

## Runtime behavior

The native batcher computes admission lengths from the padded video canvas. The model removes spatial padding after VAE encoding, so effective GEN length can be smaller than the admission count. This profile changes only the fixed budgets; it does not change admission counting or packing boundaries for a given budget.

The existing `SequencePackingPadding` training callback remains responsible for token/padding logging. Experiment-only census, replay, benchmark, audit tools and their opt-in collector have been removed; normal training does not depend on them. Historical plans and results remain local under the ignored `plans/` and `outputs/` directories.
