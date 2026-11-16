"""Opt-in Action128 integration evidence: sources, losses, and tower gradients.

This diagnostic callback adds synchronization and is not a throughput benchmark.
It does not change model inputs, RNG state, or gradients.
"""

import hashlib
import json

import torch
import torch.distributed as dist

from cosmos_framework.callbacks.small_action_audit import SmallActionAudit


def _scalars(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().flatten().tolist()
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _scalars(child)]
    return [value]


class Action128Audit(SmallActionAudit):
    def __init__(self, expected_global_batch_size: int = 8):
        super().__init__(expected_global_batch_size)

    def on_train_start(self, model, iteration=0):
        super().on_train_start(model, iteration)
        model.net.action_normalization.validate_contract()
        self.registry = model.config.action128_spec["registry"]
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.log_path = self.output / f"action128_rank_{self.rank}.jsonl"
        self._write_head_state(model, iteration, "start")
        self._spatial_hook = model.net.register_forward_pre_hook(self._audit_spatial_inputs, with_kwargs=True)

    def _write_head_state(self, model, iteration, phase):
        checksums = {}
        for name, parameter in model.net.named_parameters():
            if not name.startswith(("action2llm.", "llm2action.")):
                continue
            value = parameter.detach()
            if hasattr(value, "to_local"):
                value = value.to_local()
            raw = value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            checksums[name] = hashlib.sha256(raw).hexdigest()
        record = {
            "iteration": iteration,
            "rank": self.rank,
            "heads": checksums,
            "vision_spatial_position_mode": model.net.action_normalization.vision_spatial_position_mode,
            "contract_digest": model.net.action_normalization.contract_digest.cpu().tolist(),
        }
        path = self.output / f"{phase}_iter_{iteration:09d}_rank_{self.rank}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")

    def _audit_spatial_inputs(self, module, args, kwargs):
        packed = kwargs.get("packed_seq", args[0] if args else None)
        mode = module.config.vision_spatial_position_mode
        if packed.vision_spatial_position_mode != mode:
            raise RuntimeError("Model received a mismatched spatial RoPE mode")
        items = []
        offset = 0
        seams = packed.spatial_seams_vision or [None] * len(packed.vision.token_shapes)
        for shape, seam in zip(packed.vision.token_shapes, seams, strict=True):
            frames, height, width = shape
            count = frames * height * width
            indexes = packed.vision.sequence_indexes[offset : offset + count]
            ids = packed.position_ids[:, indexes].reshape(3, frames, height, width)
            expected_h = torch.arange(height, dtype=torch.float32, device=ids.device)
            if seam is not None:
                expected_h = seam - expected_h
            expected_h = expected_h[None, :, None].expand(frames, height, width)
            expected_w = torch.arange(width, dtype=torch.float32, device=ids.device)[None, None].expand(
                frames, height, width
            )
            torch.testing.assert_close(ids[1].float(), expected_h, rtol=0, atol=0)
            torch.testing.assert_close(ids[2].float(), expected_w, rtol=0, atol=0)
            items.append(
                {
                    "grid_thw": list(shape),
                    "seam_tokens": seam,
                    "height_first": float(ids[1, 0, 0, 0]),
                    "height_last": float(ids[1, 0, -1, 0]),
                    "negative_height_tokens": int((ids[1] < 0).sum()),
                }
            )
            offset += count
        self.spatial_evidence = {"mode": mode, "items": items}

    def on_train_end(self, model, iteration=0):
        self._spatial_hook.remove()
        self._write_head_state(model, iteration, "end")

    def on_after_backward(self, model, iteration=0):
        norms = None
        for name, parameter in model.net.named_parameters():
            if parameter.grad is None:
                continue
            if not parameter.requires_grad:
                raise RuntimeError(f"Frozen parameter received a gradient: {name}")
            grad = parameter.grad
            if hasattr(grad, "to_local"):
                grad = grad.to_local()
            if norms is None:
                norms = torch.zeros(3, device=grad.device)
            is_action = any(
                key in name
                for key in (
                    "moe_gen_action",
                    "action_time_embedder",
                    "action2llm",
                    "llm2action",
                    "action_modality_embed",
                )
            )
            is_video = any(key in name for key in ("moe_gen", "time_embedder", "vae2llm", "llm2vae"))
            role = 2 if is_action else 1 if is_video else 0
            norms[role] += grad.float().square().sum()
        if norms is None:
            raise RuntimeError("No gradients were produced")
        if dist.is_initialized():
            dist.all_reduce(norms)
        if not torch.isfinite(norms).all():
            raise RuntimeError("Non-finite Action128 gradients before clipping")
        if norms[0] != 0 or norms[1] <= 0 or norms[2] <= 0:
            raise RuntimeError(f"Expected frozen UND and active VIDEO/ACTION gradients, got {norms.tolist()}")
        self.gradient_norms = dict(zip(("UND", "VIDEO", "ACTION"), norms.sqrt().tolist(), strict=True))

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        metrics = {"loss": float(loss.detach().float().item())}
        for key in ("flow_matching_loss_vision", "flow_matching_loss_action"):
            metrics[key] = float(output_batch[key].detach().float().item())
        if not all(torch.isfinite(torch.tensor(value)).item() for value in metrics.values()):
            raise RuntimeError(f"Non-finite Action128 loss: {metrics}")
        normalization_ids = [int(value) for value in _scalars(data_batch["normalization_domain_id"])]
        record = {
            "iteration": iteration,
            "rank": self.rank,
            "dataset": data_batch["dataset_name"],
            "sample_keys": _scalars(data_batch["__key__"]),
            "projection_ids": _scalars(data_batch["domain_id"]),
            "normalization_ids": normalization_ids,
            "normalization_domains": [self.registry["domains"][index] for index in normalization_ids],
            "tokens": {
                key: int(output_batch[key])
                for key in ("und_token_length", "vision_token_length", "action_token_length")
            },
            "gradient_norms": self.gradient_norms,
            "spatial_rope": self.spatial_evidence,
            "camera_layouts": [item["camera_layout"] for item in data_batch["decode_metadata"]],
            **metrics,
        }
        with self.log_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
