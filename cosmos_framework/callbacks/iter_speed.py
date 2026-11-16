# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import time
from typing import Any

import torch
import wandb
from torch import Tensor

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import rank0_only
from cosmos_framework.utils.easy_io import easy_io


def _vfm_batch_size_metrics(model: ImaginaireModel, data_batch: dict[str, Any]) -> dict[str, int]:
    """Return the legacy image/video batch-size metric without assuming a DiT target key.

    Generation batches carry a top-level ``images`` or ``video`` target. Packed
    Understanding batches instead carry processed media in ``vlm_inputs``. The
    model classifier already understands both representations, so use it for
    the metric bucket and fall back to ``vlm_inputs`` for the value being
    counted. Text-only batches may have neither representation; they simply do
    not emit a media batch-size metric.
    """
    is_image_batch = model.is_image_batch(data_batch)
    metric_name = "image_batch_size" if is_image_batch else "video_batch_size"
    if is_image_batch:
        media_key = model.input_image_key
    else:
        input_video_latent_key = getattr(model, "input_video_latent_key", None)
        media_key = (
            input_video_latent_key
            if input_video_latent_key is not None and input_video_latent_key in data_batch
            else model.input_video_key
        )
    media_batch = data_batch.get(media_key)
    if media_batch is None:
        media_batch = data_batch.get("vlm_inputs")
    if media_batch is None:
        return {}
    return {metric_name: len(media_batch)}


class IterSpeed(EveryN):
    """
    Args:
        hit_thres (int): Number of iterations to wait before logging.
        save_s3 (bool): Whether to save to S3.
        save_s3_every_log_n (int): Save to S3 every n log iterations, which means save_s3_every_log_n n * every_n global iterations.
    """

    def __init__(self, *args, hit_thres: int = 5, save_s3: bool = True, save_s3_every_log_n: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.time = None
        self.hit_counter = 0
        self.hit_thres = hit_thres
        self.save_s3 = save_s3
        self.save_s3_every_log_n = save_s3_every_log_n
        self.name = self.__class__.__name__
        self.last_hit_time = time.time()

    def on_training_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        if self.hit_counter < self.hit_thres:
            batch_metrics: dict[str, int] = {}
            if (
                hasattr(model, "is_image_batch")
                and hasattr(model, "input_image_key")
                and hasattr(model, "input_video_key")
            ):
                batch_metrics = _vfm_batch_size_metrics(model, data_batch)
            batch_summary = "".join(f" | {name}: {value}" for name, value in batch_metrics.items())
            log.info(
                f"Iteration {iteration}: "
                f"Loss: {loss.detach().item():.4f} | "
                f"Time: {time.time() - self.last_hit_time:.2f}s | "
                f"Warmup: {self.hit_counter + 1}/{self.hit_thres}"
                f"{batch_summary}",
                rank0_only=False,
            )
            self.hit_counter += 1
            self.last_hit_time = time.time()
            #! useful for large scale training and avoid oom crash in the first two iterations!!!
            torch.cuda.synchronize()
            return
        super().on_training_step_end(model, data_batch, output_batch, loss, iteration)

    @rank0_only
    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, Tensor],
        output_batch: dict[str, Tensor],
        loss: Tensor,
        iteration: int,
    ) -> None:
        if self.time is None:
            self.time = time.time()
            return
        cur_time = time.time()
        iter_speed = (cur_time - self.time) / self.every_n / self.step_size

        log.info(
            f"Iteration {iteration}: Loss: {loss.detach().item():.4f} | "
            f"Iter speed: {iter_speed:.2f} seconds per iteration",
        )

        per_sample_batch_counter = dict()
        # for VFM
        if hasattr(model, "is_image_batch") and hasattr(model, "input_image_key") and hasattr(model, "input_video_key"):
            per_sample_batch_counter.update(_vfm_batch_size_metrics(model, data_batch))
        # for LLM training only
        elif "input_ids" in data_batch:
            mbs = data_batch["input_ids"].shape[0]
            dp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            grad_accum_iter = int(trainer.config.trainer.grad_accum_iter)
            per_sample_batch_counter["token_batch_size"] = mbs
            per_sample_batch_counter["token_global_batch_size"] = mbs * dp_size * grad_accum_iter
            # Cumulative token count (LLM analog of sample_counter). Set by
            # ``LLMPretrainModel.training_step`` into a persistent buffer on
            # ``model.net``, so this value survives checkpoint resume.
            if hasattr(model, "token_counter"):
                per_sample_batch_counter["token_counter"] = model.token_counter

        if wandb.run:
            sample_counter = getattr(trainer, "sample_counter", iteration)
            wandb.log(
                {
                    "timer/iter_speed": iter_speed,
                    "sample_counter": sample_counter,
                }
                | per_sample_batch_counter,
                step=iteration,
            )
        self.time = cur_time
        if self.save_s3:
            if iteration % (self.save_s3_every_log_n * self.every_n) == 0:
                easy_io.dump(
                    {
                        "iter_speed": iter_speed,
                        "iteration": iteration,
                    },
                    f"s3://rundir/{self.name}/iter_{iteration:09d}.yaml",
                )
