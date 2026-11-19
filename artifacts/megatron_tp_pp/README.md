# TP/PP experiment artifacts

This directory contains local experiment evidence: generated checkpoints,
tensor samples, detailed parameter comparisons, process identities and
node-specific orchestration scripts. These files can be large and often refer
to shared checkpoints or temporary runtime paths. New generated files are
ignored; existing tracked historical reports remain tracked.

Reusable implementations and test entry points belong in `tools/` and
`tests/megatron/`. Portable isolated runtime candidates may be placed in
`candidates/*.py`.

For a deliberate delivery, select a small summary report, review its contents,
and add that exact file with `git add -f <path>`. Do not force-add the whole
experiment directory. Full runtime evidence remains available in the original
shared workspace and under `outputs/`; a Git checkout does not contain those
large training outputs.

Current implementation status and acceptance limits are documented in
[the TP/PP implementation report](../../docs/megatron_tp_pp.md).
