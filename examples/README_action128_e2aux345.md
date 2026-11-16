# Action128 initialization from E2aux iter345

`launch_sft_action128_wam_nano_160gpu.sh` now defaults to:

```
/share/project/eai_pwm/models/Cosmos3-Nano-E2aux-treated38qe1-iter345-action1b/width_only_1b
```

This is a model-only initialization from the regular `net.*` weights in:

```
/share/project/eai_pwm/action_line/runs/cosmos3_action_fd/action_sft/E2aux_treated38qe1_from1150/checkpoints/iter_000000345
```

The inference export `E2aux_treated38qe1_from1150_iter345_hf_ema_novit` uses EMA and is not the source of this conversion. Optimizer, scheduler, iteration, and dataloader states are not imported.

## Conversion

The source is a two-tower, 64D-action checkpoint. Preserve its UND/VIDEO weights and derive a `width_only_1b` ACTION tower from its trained GEN weights using the existing deterministic channel-selection and fan-in scaling algorithm. The target has 36 ACTION layers, hidden size 1280, FFN size 3840, and 1,010,059,264 action-related parameters before Action128 head replacement.

The source lacks five sound tensors found in original Nano. Fill only those from `/share/project/eai_pwm/models/Cosmos3-Nano-DCP`; the current WAM recipe disables sound generation. Missing non-sound tensors are rejected rather than silently filled. EMA tensors are excluded.

The current Action128 contract differs from the source: `action128_load_mode=auto` initializes new `action2llm`, `llm2action`, and normalization state. The old 64D action heads are not reused as Action128 heads.

Reproduce with a new output directory:

```bash
LD_LIBRARY_PATH='' PYTHONPATH=. /opt/venv/bin/python tools/prepare_small_action_checkpoint.py \
  --source /share/project/eai_pwm/action_line/runs/cosmos3_action_fd/action_sft/E2aux_treated38qe1_from1150/checkpoints/iter_000000345 \
  --fallback-source /share/project/eai_pwm/models/Cosmos3-Nano-DCP \
  --output /share/project/eai_pwm/models/Cosmos3-Nano-E2aux-treated38qe1-iter345-action1b \
  --presets width_only_1b
```

Existing output directories are rejected. Conversion provenance is saved in `conversion.json`; the artifact's `validation.json` records full tensor verification. Diagnostic scripts/logs live in `outputs/action128_e2aux345_init/`.

## Launch

The existing launcher and TOML filenames are unchanged. The new job name includes `e2aux345` and the default output root is `outputs/action128_wam_e2aux345_160gpu_static_vae_aot_no_robomind2_run01`, preventing automatic recovery from the old run4 namespace.

Explicit environment variables still override defaults. Remove an inherited old `BASE_CHECKPOINT_PATH`, or explicitly export the new path above. Use a fresh `OUTPUT_ROOT` for an independent experiment. Subsequent restarts with this same job/output will resume its latest checkpoint normally.

The original Nano two-tower baseline launcher remains on `Cosmos3-Nano-DCP`. It is therefore an original-model baseline, not an experiment with the same E2aux initialization. Current three-tower LR/GA remain `1e-4`/2; the existing two-tower recipe retains `2e-5`/1.

Validation includes training CLI dry-run, conversion regression tests, target key/shape comparison with the previous three-tower initialization, and a full reload compared exactly against the converted source tensors. No new GPU training run was launched for this change.
