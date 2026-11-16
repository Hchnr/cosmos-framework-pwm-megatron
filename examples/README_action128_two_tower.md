# Action128 original Nano two-tower baseline

**Current production update:** the three-tower launcher now starts from E2aux iter345 and uses LR `1e-4` / GA2. This original Nano baseline retains LR `2e-5` / GA1. The matching description below records its initial setup; see [current initialization](README_action128_e2aux345.md).

Run `bash examples/launch_sft_action128_wam_nano_2tower_160gpu.sh` on every node using the same platform job setup as `launch_sft_action128_wam_nano_160gpu.sh`. The paired recipe is `toml/sft_config/action128_wam_nano_2tower_160gpu.toml`.

The default is 20 × 8 GPUs, shard16. Set `NNODES` to match the three-tower experiment exactly; for an 18-node comparison, use `NNODES=18` on both jobs. The wrapper reuses the production launcher's rendezvous, THP policy, seed42 and overrides. Its default port is 50013.

## Initialization and matching

The two-tower checkpoint defaults to `/share/project/eai_pwm/models/Cosmos3-Nano-DCP` (the DCP root, not its `model/` subdirectory). VIDEO and ACTION share the original GEN tower. Do not inherit `BASE_CHECKPOINT_PATH` from the three-tower job: environment overrides take precedence over defaults.

The recipes match the 12 source catalogs and weights (excluding RoboMIND2.0), FPS policy, WAM rendering, normalization, workers, admission limits, LR/scheduler, GA1, shard16, BF16, full activation checkpointing, static language compilation, strict VAE AOT, and checkpoint policy. Here “admission limits” refers to total UND/GEN budgets; the ACTION partition differs as described below.

Necessary differences:

- `two_tower`, preset `full`, original Nano checkpoint, shared GEN modules; no separate 1B ACTION tower.
- Original `auto` / `split_default` attention instead of the small three-tower `flash3_video` backend.
- No small-tower UND short bucket: UND padding remains 11776, GEN remains 62592.
- No independent `action_tokens=2944` partition: two-tower supports shared GEN padding only. This also removes the loader's separate VIDEO/ACTION admission caps, so individual packed batches can differ even with identical catalogs and seed.
- Independent job name, default output directory, compile-cache directory and port.

Job name: `action128_wam_nano_2tower_12src_no_robomind2_160gpu_videocap30_static_vae_aot`.

Default output root: `outputs/action128_wam_2tower_160gpu_static_vae_aot_no_robomind2_run01`. Use a fresh output root when starting an independent experiment; the training launcher automatically resumes checkpoints in an existing run directory. Explicit platform `OUTPUT_ROOT` overrides win.

Compare `train/loss`, `train@2_detail/flow_matching_loss_action`, and `train@2_detail/flow_matching_loss_vision` over sufficiently long windows, at the same GPU count and preferably grouped by data source. This is an original-model baseline, not an identical-initialization or identical-batch ablation.

## Historical reference and validation

Git commit `50b2762` introduced `examples/launch_sft_action_policy_libero_single_task_two_tower.sh` and its paired TOML; they were removed during later production-entry consolidation (`995be5b`). That recipe used the original Nano checkpoint and shared GEN training, but was a LIBERO single-task experiment rather than this online mixture. Its W&B mode was disabled.

The earlier local record also documents a successful four-GPU, two-step `nano_two_tower_smoke` checkpoint. These historical checks establish a previous two-tower training path, not validation of this new multi-source configuration.

Current validation: production training CLI `--dryrun` completed; schema, shell syntax and matching common recipe fields checked. No new GPU training or online launch was performed for this configuration change. Dry-run output: `outputs/action128_two_tower_config_review/dryrun.log`.

## Checkpoint retention

Both Action128 recipes set `[checkpoint].keep_last_n = 3`. After all ranks finish saving a new checkpoint and publish `latest_checkpoint.txt`, rank zero removes older completed `iter_*` directories in that job's checkpoint directory. Async saves apply the same policy inside the background worker after writing, never when enqueuing a save. Failed saves do not trigger cleanup.

The three-tower production recipe also sets `keep_every_n_steps = 2000`: steps 2000, 4000, ... survive cleanup in addition to the newest three checkpoints. The interval must be a positive multiple of `save_iter` (currently 200). For example, after saving step 4600, completed steps 2000, 4000, 4200, 4400 and 4600 remain. The two-tower recipe retains its newest-three-only policy. Existing completed checkpoints obey the updated policy on the next successful save after restart; checkpoints already deleted cannot be recovered.

Retention supports local/shared filesystem DCP storage. Omit `keep_last_n` to keep all checkpoints; enabling it with object-store saves is rejected. Incomplete directories, symlinks and checkpoints newer than the just-saved step are left untouched, so the total number of directories can exceed three. Older checkpoints without a completion marker are eligible only when all four model/optimizer/scheduler/trainer DCP metadata files exist. Deletion failures are logged and retried on a later successful save.
