# Bounded Action128 camera decoding

`UnifiedAction128Dataset` can overlap camera decoding within one sample for
explicitly selected dataset roots:

```bash
export COSMOS_ACTION128_PARALLEL_VIEW_SOURCES=egopro
export COSMOS_DATALOADER_TIMING=1
```

Set these before starting training so loader workers and decode helpers inherit
them. The default is serial decoding. Remove the first variable to disable the
candidate. Changing a workspace file or setting a variable in another shell does
not change an already running training release.

The source list contains exact root directory basenames, separated by commas.
Only two- or three-camera MP4/MCAP samples with an output canvas at most 512 MiB
are eligible. Other samples retain the serial path. Each camera has an independent
helper/socket/codec; at most two requests execute concurrently. A three-camera
sample uses at most three helpers, two more than its serial counterpart. Helpers
are reused across windows. Existing per-helper safety limits still apply; the
512 MiB canvas limit does not cap decoder internals or total process memory.

Timeline resolution stays on the calling thread. Results are copied in the
original camera order. Logical streams, sampling, captions, packing, RNG,
checkpoint layout, physical workers and global prefetch credits are unchanged.
A media failure joins outstanding requests before resetting every camera helper.

The target is serial preparation backlog after background receiving has already
removed result-send blocking. On the current production run, step 7039 took
121.69 seconds: rank 146 prepared three consecutive EgoPro batches in roughly
66–69 seconds each, with the latter two consumer waits at 33 and 58 seconds.
Their transfer times were below one second. More background receiving cannot
remove this preparation work.

Regression coverage lives in `cosmos_framework/data/test/parallel_video_test.py`.
Production rollout requires exact-window pixel/metadata comparisons, process-tree
memory measurements, and a representative loader/resume comparison. Video-only
CPU replay improvements do not establish a GPU step-time improvement or prove
that all long tails are eliminated.

## CPU replay results (2026-11-04)

Sequential serial/parallel/serial runs replayed the exact EgoPro episode 107137,
windows 25–66 from the long step above. The 42-window video totals were 146.54 /
86.20 / 146.76 seconds. Excluding the first window, the warm serial recheck fell
from 141.36 to 80.86 seconds (42.8%). All pixel, validity-mask, and non-timing
metadata hashes matched. Sampled process-tree PSS peaked at 1.64 / 2.58 / 1.64
GiB; the candidate used three helpers instead of one. This is a single pipeline,
not a full-node memory estimate.

EgoStandard episode 196035 windows 24, 25, 26, 40, 41 also matched, including
tail padding. Its warm video time fell from 6.557 to 5.025 seconds (23.4%). This
does not establish the roughly 26.6% whole-batch preparation improvement needed
by the previously observed dense EgoStandard segment. Merely increasing MCAP
decoder threads from one to four improved a separate short pilot by only 0.6%;
that experiment was not integrated.

Evidence is stored under `outputs/slowstep_20261104/view_parallel/`:
`comparison_aba.json`, `edge_comparison.json`, and
`mcap_threads_pilot_summary.json`. These artifacts are ignored by Git.

The real 58-source loader also passed a same-parent serial/parallel/serial replay
from the complete step-7400 checkpoint (rank 146, 42 microbatches, W16/K8/BGRecv).
All complete-batch hashes and final committed cursor/RNG hashes matched. Total
CPU replay times, including fingerprinting and with no simulated compute, were
190.05 / 157.51 / 175.17 seconds (10.1% reduction versus the serial recheck).
This is not a GPU throughput measurement. See `loader_resume_comparison.json`.
