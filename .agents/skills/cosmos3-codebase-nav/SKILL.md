---
name: cosmos3-codebase-nav
description: >
  Navigate the current Cosmos3 repository to locate training and inference entry points,
  TOML configuration, Parquet dataflow, Qwen3-VL processing, task-specific batch contracts,
  model consumers, recipes, defaults, and tests. Use for questions about where code or
  configuration lives and how a configured task reaches its runtime implementation.
---

# Cosmos3 Codebase Navigation

Use the repository's current files as the source of truth. Read `../../../AGENTS.md`
first for the top-level map and commands; use this skill for the non-obvious routing
between configuration, data, and model code.

All paths below are relative to this skill directory
(`.agents/skills/cosmos3-codebase-nav/`).

## Training entry and configuration

The supported SFT entry point is:

```text
examples/launch_sft_*.sh
  -> python -m cosmos_framework.scripts.train --sft-toml=<recipe>
  -> cosmos_framework/configs/toml_config/sft_config.py
  -> selected Hydra experiment/model plus the configured dataloader
```

Key locations:

| What | Where |
| --- | --- |
| Training CLI | `../../../cosmos_framework/scripts/train.py` |
| TOML validation and config assembly | `../../../cosmos_framework/configs/toml_config/sft_config.py` |
| Maintained SFT TOMLs | `../../../examples/toml/sft_config/` |
| Matching launchers | `../../../examples/launch_sft_*.sh` |
| VFM, Action, FD/ID/WAM experiments | `../../../cosmos_framework/configs/base/experiment/` |
| Standalone VLM experiments | `../../../cosmos_framework/configs/base/reasoner/experiment/` |
| Shared model defaults | `../../../cosmos_framework/configs/base/defaults/` |
| Standalone VLM defaults | `../../../cosmos_framework/configs/base/reasoner/defaults/` |

`configs/base/reasoner/` names the standalone VLM configuration branch. It does
not imply that a separate Reasoner data processor still exists.

## Current training data path

When `[[dataloader_train.datasets]]` is present, the TOML loader replaces the
recipe's legacy dataloader with the catalog path:

```text
five-column Parquet catalog
  -> ParquetCatalogDataset / ParquetBlockDistributor
  -> task-specific RawItemProcessor
  -> task-specific SampleBatcher
  -> task-specific BatchCollator
  -> JointCosmosDataLoader
```

The five catalog columns are `id / type / dataset / data / provenance`.

| Layer | Where | Responsibility |
| --- | --- | --- |
| Schema, footer index, row-group access, shuffle | `../../../cosmos_framework/data/generator/catalog.py` | Lazy Parquet reads and distributed row ordering |
| Task rendering and dataloader factory | `../../../cosmos_framework/data/generator/catalog_processors.py` | Understanding, visual generation, and Action catalog dispatch |
| Four-role interfaces | `../../../cosmos_framework/data/generator/dataflow/base.py` | Distributor, processor, batcher, collator contracts |
| Distributors | `../../../cosmos_framework/data/generator/dataflow/distributors.py` | Iterable/map-style rank and worker partitioning |
| Batchers | `../../../cosmos_framework/data/generator/dataflow/batchers.py` | Pool or sequential token-budget grouping |
| Collators | `../../../cosmos_framework/data/generator/dataflow/collators.py` | VLM padded batches and VFM legacy-compatible nested batches |
| Loader orchestration and ratios | `../../../cosmos_framework/data/generator/dataflow/loader.py` | CosmosDataLoader and JointCosmosDataLoader |
| Media materialization | `../../../cosmos_framework/data/generator/media_decode.py` | Bounded video decoding, timestamps, retry, and helper lifecycle |
| Checkpointed loader state | `../../../cosmos_framework/callbacks/cosmos_dataloader_state.py` | Joint schedule and stream cursor save/restore |

Do not mistake `../../../cosmos_framework/data/generator/joint_dataloader.py`
for the current Parquet entry point. It is retained for compatibility with older
callers and golden tests. New catalog training is assembled by
`build_catalog_training_dataloader()` in `catalog_processors.py`.

## Processor and task contracts

The only public training processor is `Qwen3VLProcessor` in
`../../../cosmos_framework/data/generator/processors/qwen3vl_processor.py`.
Nano and Edge both use that class. Edge-specific SigLIP2 media construction is a
private backend in
`../../../cosmos_framework/data/generator/processors/cosmos3_edge_processing.py`;
it is not a second chat-template or loss-mask implementation.

The catalog path deliberately has two model-facing batch contracts:

| Task | Processor/collator output | Model consumer |
| --- | --- | --- |
| Understanding | Padded `input_ids`, `attention_mask`, `labels`, and Qwen media tensors | `../../../cosmos_framework/model/generator/vlm_model.py` |
| T2I/T2V/TI2V/V2V | Legacy-compatible nested `text_token_ids`, `images`/`video`, `sequence_plan`, metadata | `../../../cosmos_framework/model/generator/omni_mot_model.py` |
| Action, FD, ID, WAM | Nested text/video/action plus `sequence_plan` | `../../../cosmos_framework/model/generator/omni_mot_model.py` |

Understanding uses padded HF-style sequences. Generation and Action are packed
inside OmniMoT after VAE/action encoding by
`../../../cosmos_framework/data/generator/sequence_packing/`; do not infer a
format bug merely because their raw batches differ.

Generation chat rendering and assistant-text supervision live in
`CatalogVisualGenerationProcessor`. Target-prefix conditioning is represented by
`SequencePlan.condition_frame_indexes_vision`, not by a separate source video.

## Action data

An Action catalog Parquet represents exactly one LeRobot dataset and contains one
catalog row. Its `data.root` points to the dataset; episodes and training windows
are expanded dynamically.

| What | Where |
| --- | --- |
| Action dataset factories and modes | `../../../cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` |
| LeRobot implementations | `../../../cosmos_framework/data/generator/action/datasets/` |
| Video/state/action transforms and SequencePlan | `../../../cosmos_framework/data/generator/action/utils/transforms.py` |
| Action normalization | `../../../cosmos_framework/data/generator/action/action_normalization.py` |
| Checked-in normalizer statistics | `../../../cosmos_framework/data/generator/action/normalizer_stats/` |

Keep Forward Dynamics, Inverse Dynamics, Action Policy, and WAM distinctions when
following Action code; they share infrastructure but use different condition and
loss masks.

## Tests that establish data invariants

All data tests are under `../../../cosmos_framework/data/test/`.

| Invariant | Test |
| --- | --- |
| Parquet schema, task rendering, conditioning, and mixed ratios | `catalog_test.py` |
| New VFM collator versus legacy batch nesting | `golden_vfm_test.py` |
| Real Qwen3-VL chat and assistant masks | `qwen3vl_processor_test.py` |
| Edge media backend | `cosmos3_edge_processing_test.py` |
| Deterministic resume | `resume_test.py` |
| Media limits, seek, timeout, and retry | `media_decode_test.py` |
| Action transform semantics | `transforms_test.py` |

Structural equality in `golden_vfm_test.py` starts from already processed samples;
it does not by itself prove chat-template or final packed-token equality.

## Inference is separate

Training-time imports should not be added to inference code, or vice versa.

| What | Where |
| --- | --- |
| Inference CLI | `../../../cosmos_framework/scripts/inference.py` |
| Arguments and override resolution | `../../../cosmos_framework/inference/args.py` |
| Per-mode defaults | `../../../cosmos_framework/inference/defaults/<mode>/sample_args.json` |
| Core setup and sampling | `../../../cosmos_framework/inference/model.py`, `../../../cosmos_framework/inference/inference.py` |
| Ray serving | `../../../cosmos_framework/inference/ray/` |
| Example inputs | `../../../inputs/omni/`, `../../../inputs/reasoner/` |
| Backend shims | `../../../packages/{diffusers,transformers,vllm}-cosmos3/` |

The remaining JSON/JSONL conversion scripts under
`../../../cosmos_framework/scripts/` are not the current Parquet training input
path. Confirm callers before treating one as dead code because some serve
inference, evaluation, or backward compatibility.

## Navigation method

Prefer targeted searches over assuming an older directory layout:

```bash
rg -n "symbol_or_config_key" cosmos_framework examples docs
rg --files cosmos_framework/data cosmos_framework/configs examples
git grep -n "symbol_or_config_key"
```

For a configuration value, follow it in this order:

```text
recipe TOML
  -> pydantic field/default in sft_config.py
  -> config assembly/override
  -> dataloader or model constructor
  -> runtime use
  -> focused test
```

For comparisons with the released/original implementation, inspect the intended
Git ref explicitly (for example `git show upstream/main:<path>`); do not call the
current working tree "original Cosmos".
