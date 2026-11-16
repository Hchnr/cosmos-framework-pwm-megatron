# Cosmos3 SFT Examples

Runnable artifacts for Cosmos3 supervised fine-tuning. The end-to-end walkthrough — data preparation, base-checkpoint conversion, launch, outputs, export to safetensors, and evaluation — lives in **[docs/training.md](../docs/training.md)**. Start there.

This directory contains:

- `toml/sft_config/` — TOML recipes consumed by `cosmos_framework.scripts.train --sft-toml=…`. One file per recipe. The TOML is validated against the pydantic schema at [`cosmos_framework/configs/toml_config/sft_config.py`](../cosmos_framework/configs/toml_config/sft_config.py) at load time.
- `launch_sft_*.sh` — paired launch shells. Each declares `TOML_FILE` plus `: "${DATASET_PATH:=…}"` / `: "${BASE_CHECKPOINT_PATH:=…}"` defaults (full repo-relative paths, matching what [`docs/training.md`](../docs/training.md) shows) and sources [`_sft_launcher_common.sh`](./_sft_launcher_common.sh), which sets the `torchrun` flags and forwards into `cosmos_framework.scripts.train`. `export`ing those vars in your shell before launching wins over the defaults; otherwise just run the shell after Steps 1+2 of `docs/training.md`.
- `inference.py`, `inference_pipeline.py` — runnable inference helpers; see [docs/inference.md](../docs/inference.md).

## 58-source State-H1 production continuation

Use [`launch_pretrain_state_h1_58src_online.sh`](launch_pretrain_state_h1_58src_online.sh) for the current 20-node continuation, including container environment setup and a new SwanLab curve. See [configuration and usage](README_pretrain_state_h1_online.md).

## Recipe → launch shell

| Recipe                                       | Launch shell                          |
| -------------------------------------------- | ------------------------------------- |
| Action policy, production scheme D (~1B action tower) | `launch_sft_action_policy_1b.sh` |
| Action128 12-source WAM, excluding RoboMIND2.0 (~1B ACTION tower) | `launch_sft_action128_wam_nano_160gpu.sh` |
| Action128 matched original Nano two-tower loss baseline | `launch_sft_action128_wam_nano_2tower_160gpu.sh` |
| Vision SFT (Cosmos3-Nano)                    | `launch_sft_vision_nano.sh`           |
| Vision SFT (Cosmos3-Edge)                    | `launch_sft_vision_edge.sh`           |
| Reasoner Alignment SFT                       | `launch_sft_llava_ov.sh`              |
| Reasoner Alignment SFT (Cosmos3-Nano)        | `launch_sft_videophy2_nano.sh`        |
| Reasoner Alignment SFT (Cosmos3-Edge)        | `launch_sft_videophy2_edge.sh`        |

For the multi-node USR offline-VAE experiment and 8-GPU smoke overrides, see [README_usr_v1.md](README_usr_v1.md).

For the production action policy, including local1×8 and multi-node8-GPU-per-host
launches, Offline VAE preparation, explicit global batch sizing, converted checkpoints and local CLI overrides, see
[docs/action_policy_1b.md](../docs/action_policy_1b.md).

## Action128 two-tower loss baseline

See [README_action128_two_tower.md](README_action128_two_tower.md) for the paired configuration, original checkpoint, and comparison limits.
