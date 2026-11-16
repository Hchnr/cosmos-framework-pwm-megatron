# Selective deep media lookahead (experimental, default off)

Global `prefetch_factor=8` continues to bound ordinary model-facing batches.
An additional, disposable media cache can look far ahead in selected Action128
streams without preparing every future batch or advancing a consumed cursor.

```bash
export COSMOS_ACTION128_MEDIA_PREFETCH_SOURCES=egopro,egostandard
export COSMOS_ACTION128_MEDIA_LOOKAHEAD=1024
export COSMOS_ACTION128_MEDIA_PREFETCH_TOTAL_BYTES=2147483648
export COSMOS_ACTION128_MEDIA_PREFETCH_BYTES=2147483648
export COSMOS_ACTION128_MEDIA_PREFETCH_SOURCE_BYTES='{"egopro":536870912,"egostandard":1610612736}'
export COSMOS_ACTION128_MEDIA_MIN_PIXEL_WORK=100000000
```

The source list uses exact root directory basenames. Set these before launching
training. Remove `COSMOS_ACTION128_MEDIA_PREFETCH_SOURCES` to disable the feature.
The byte total is shared atomically by every physical worker of one rank, rather
than multiplied by the number of workers. The per-stream cap can be overridden
by root name with `COSMOS_ACTION128_MEDIA_PREFETCH_SOURCE_BYTES`, a JSON object.
When selecting multiple sources, set their quotas and the rank total together.
The shared rank budget is integrated with `SharedPrefetchDataLoader` using
physical workers (`num_workers > 0`); other loader backends are not validated.

The lifecycle is:

1. At worker startup, only sources assigned to that worker build lookahead.
   Candidate enumeration uses the original episode order, DP/virtual shard and
   resume position. It reads resume variables without consuming them.
2. Cheap plans inspect camera resolution, frame counts, view count and storage.
   MCAP cost includes its current three-second pre-roll. This is a structural
   ranking heuristic, **not** an exact GOP or I/O-latency predictor. A plan window
   of 1024 means metadata entries, not 1024 decoded windows.
3. Only selected, sufficiently expensive MP4/MCAP windows acquire output-byte
   reservations. Background materialization reads projected timestamp/PTS rows,
   excluding Action and State arrays, then decodes the original video path.
4. The normal foreground sampler, numeric State/Action processing, caption,
   augmentation, packing and checkpoint commit remain in their original order.
   A hit supplies video only. Misses, failed speculation and oversized outputs
   use the ordinary foreground path; background failures never trigger skipping.

Reservations include running and ready **output tensors** and transfer to the
ordinary batch window when consumed. They do not cap codec workspaces, projected
Arrow buffers, Python metadata or the existing K8 batch window. Total memory must
therefore also be checked with process-tree PSS and node memory measurements.
Only enable this candidate on immutable training inputs; cached decoded windows
are not an update mechanism for datasets modified during a run.

The cache is absent from checkpoint state and can be discarded on restart. The
foreground still applies the original media failure/exclusion policy when a
window is actually consumed. On a source timeout, its iterator closes its cache
before the replacement stream is built.

## CPU replay validation (2026-11-04)

Real 58-source loader, full checkpoint 7400/rank 146, 42 microbatches from the
same initialized parent, W16/K8/BGRecv, fixed three-second simulated compute
including fingerprint time. Only B enables the two-source cache above; camera
parallelism remains off. A/B/A elapsed times: 194.805 / 160.342 / 196.119 seconds.

| Metric | Serial C | Selective B |
| --- | ---: | ---: |
| Total replay | 196.119 s | 160.342 s (−18.24%) |
| Sum of fetch waits | 69.240 s | 33.415 s |
| EgoStandard wait, microbatch 34 | 11.575926 s | 0.000060 s |
| EgoStandard wait, microbatch 36 | 14.361511 s | 0.000087 s |
| EgoStandard wait, microbatch 40 | 8.797784 s | 0.000061 s |
| Sampled process-tree peak PSS | 22.084 GiB | 24.036 GiB |

A's sampled PSS was 22.448 GiB: the observed additional memory is approximately
1.6–2.0 GiB. This is one-second sampling of the CPU process tree, including the
metadata parent, fork child and physical workers, not production node memory.
The output reservation peak was 2,085,865,866 bytes, below the shared 2 GiB cap.
All complete batch tensor hashes, sample/source IDs and committed cursor/RNG
states match across all three runs. An earlier EgoPro-only control improved
just 1.23% because that source already had negligible waits in this window.

These results establish CPU replay benefit, not a measured GPU training speedup
or elimination of production 100-second spikes. The feature is default off and
was not deployed during the replay. Structural selection can miss unusual GOP/I/O outliers;
production validation must cover longer windows, all ranks and node memory.

Regression: 339 passed, 5 deselected across eight related test files. Deselected
cases cover GPU training/checkpoint resume and DROID integration unavailable in
this environment (TorchCodec cannot load required FFmpeg libraries). Ruff lint
and format checks pass for all seven changed Python files.

Tests: `cosmos_framework/data/test/media_prefetch_test.py`.
Final artifacts: `outputs/slowstep_20261104/selective_media_both/`, especially
`loader_resume_comparison.json`, `resources.json`, `source_manifest.json` and
`final_regression.log`. The earlier single-source control is retained under
`outputs/slowstep_20261104/selective_media/`.

## Production rollout (2026-11-04)

The user authorized a restart after CPU validation. At 13:26 CST, the 20-node
160-GPU d39afa19 job started frozen release `522255a3ef65`, with the two-source
configuration above. All 20 nodes passed source identity and checkpoint file
preflight; all 160 ranks inherited the same flags. The previous run stopped at
8028 and the complete step-8000 checkpoint restored all five state components
at 13:37:40 CST, replaying 28 completed steps. Keep W16/K8, GA3, seed42 and the
200-step checkpoint interval; camera parallelism is disabled for this rollout.

The managed node launcher remains under
`examples/_launch_pretrain_state_h1_online_node.sh`. Its `restart_plan.json`
accepts an optional `media_prefetch` object with `sources`, `lookahead`,
`total_bytes`, `stream_bytes`, `minimum_pixel_work` and `source_bytes`. Plans
without that object keep selective lookahead disabled.

New independent SwanLab experiment: [DeepMedia2G / 6spwnplj](https://swanlab.cn/@hchnr/cosmos3_action/runs/6spwnplj).
Rollout evidence is under `outputs/mainline_deepmedia20_d39afa19_20261104/`.
Full restore and resumed step 8001 (13:43:14 CST) are confirmed. First-step
initialization/compilation time is excluded from steady-state comparisons;
cloud-metric and sample-continuation checks are recorded in the rollout artifacts.
