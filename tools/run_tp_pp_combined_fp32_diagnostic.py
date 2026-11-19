"""Opt-in full-width QKV, SDPA and row diagnostic; never production default."""

import json
import os
import sys
from pathlib import Path


def install_rows(linear_type, full_row, full_column, counts):
    previous = linear_type.forward
    if getattr(previous, "_full_row_diagnostic", False):
        raise RuntimeError("Full-row diagnostic already installed")

    def forward(self, inputs):
        if self.tp_size <= 1:
            return previous(self, inputs)
        if self.column:
            if self.bias is not None or self.weight.shape[0] * self.tp_size != self.out_features:
                raise ValueError("Combined column diagnostic requires bias-free equal shards")
            counts["mlp_column_forward_calls"] += 1
            return full_column(inputs, self.weight, self._impl.tp_group)
        if self.bias is not None:
            raise ValueError("Combined row diagnostic requires bias-free projection")
        if self.weight.shape[1] * self.tp_size != self.in_features:
            raise ValueError("Row weight does not form equal input shards")
        counts["row_forward_calls"] += 1
        counts["row_parameter_ids"].add(id(self.weight))
        return full_row(inputs, self.weight, self._impl.tp_group)

    forward._full_row_diagnostic = True
    linear_type.forward = forward


def main():
    checkout = Path(os.environ["FULL_VJP_CHECKOUT"]).resolve(strict=True)
    diag = Path("/opt/cosmos-tp-pp-full-vjp-20261119")
    attention = Path("/opt/cosmos-tp-pp-sdpa-replay-20261119")
    sys.path[:0] = [str(checkout), str(checkout / "tests/megatron"), str(diag), str(attention)]
    import importlib.util

    import tools.tp_pp_fp32_reconstructed_kv as kv
    from cosmos_framework.model.generator.megatron_tp_pp.layers import TensorParallelLinear
    from cosmos_framework.model.generator.megatron_tp_pp.stage import PipelineStage

    root = Path(__file__).resolve().parent

    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, root / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    full_row = load("_combined_row", "tp_pp_full_width_row_diagnostic.py").full_row
    native_width_column_linear = load(
        "_combined_col", "tp_pp_native_width_column_diagnostic.py"
    ).native_width_column_linear
    mark_q_projections = load("_combined_qkv", "tp_pp_expand_qkv_diagnostic.py").mark_q_projections
    install_attention = load("_combined_attention", "tp_pp_full_head_attention_experiment.py").install
    original = kv.install
    counts = {"row_forward_calls": 0, "row_parameter_ids": set(), "mlp_column_forward_calls": 0}
    attn = {"forward_calls": 0, "backward_calls": 0}

    def install(observation):
        original(observation)
        mark_q_projections(observation)
        install_attention(attn)
        install_rows(TensorParallelLinear, full_row, native_width_column_linear, counts)
        original_stage = PipelineStage.__init__

        def stage_init(stage, *args, **kwargs):
            original_stage(stage, *args, **kwargs)
            dtypes = sorted({str(p.dtype) for p in stage.parameters() if p.is_floating_point()})
            if dtypes != ["torch.float32"]:
                raise ValueError("Combined diagnostic requires FP32 stage parameters")
            counts["stage_parameter_dtypes"] = dtypes

        PipelineStage.__init__ = stage_init

    kv.install = install
    if (
        "--original-capacity" in sys.argv
        or "--attention-backend" in sys.argv
        and sys.argv[sys.argv.index("--attention-backend") + 1] != "sdpa"
    ):
        raise ValueError("Combined diagnostic supports cropped dense SDPA only")
    from run_full_model_vjp import main as run

    rank = int(os.environ["RANK"])
    report = Path(os.environ["FULL_VJP_REPORT_DIR"]) / f"combined_fp32.rank{rank}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    if report.exists():
        raise FileExistsError(report)
    error = None
    try:
        run()
        if not counts["row_forward_calls"] or not counts["mlp_column_forward_calls"] or not min(attn.values()):
            raise RuntimeError("Full-row and attention paths must both execute")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        import torch

        peak_allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
        peak_reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None
        with report.open("x") as handle:
            json.dump(
                {
                    "rank": rank,
                    "stage_parameter_dtypes": counts.get("stage_parameter_dtypes"),
                    "activation_dtype_checked": "torch.float32",
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "row_forward_calls": counts["row_forward_calls"],
                    "row_parameter_count": len(counts["row_parameter_ids"]),
                    "mlp_column_forward_calls": counts["mlp_column_forward_calls"],
                    "attention": attn,
                    "failure": error,
                    "full_model_acceptance": False,
                },
                handle,
                indent=2,
            )
            handle.write("\n")


if __name__ == "__main__":
    main()
