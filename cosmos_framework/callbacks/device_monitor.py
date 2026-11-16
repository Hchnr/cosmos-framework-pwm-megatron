# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import psutil
import pynvml
import torch
import wandb

from cosmos_framework.callbacks.every_n import EveryN
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.trainer import ImaginaireTrainer
from cosmos_framework.utils import distributed, log
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.utils.host_memory import process_tree_memory


def _log_wandb_monitor_data(data: dict[str, Any], iteration: int) -> None:
    try:
        wandb.log(data, step=iteration)
    except wandb.Error as error:
        # Monitoring uploads must not terminate the distributed training job.
        # SDK errors can contain credentials or proxy URLs; log only the type.
        log.warning(f"DeviceMonitor W&B upload failed ({type(error).__name__}); retaining local monitoring data")


def log_prof_data(
    data_list: List[Dict[str, Any]],
    iteration: int,
) -> Tuple[pd.DataFrame]:
    # Create a table to log data with rank information
    columns = ["iteration", "rank"] + list(data_list[0].keys())
    data = []

    # Initialize dictionaries to store min and max values for each metric
    min_values = {key: float("inf") for key in columns[2:]}
    max_values = {key: float("-inf") for key in columns[2:]}
    sum_values = {key: 0.0 for key in columns[2:]}
    valid_counts = {key: 0 for key in columns[2:]}

    for _rank, prof_data in enumerate(data_list):
        row = [iteration, _rank] + [prof_data[key] for key in columns[2:]]
        data.append(row)

        # Update min, max, and sum values
        for key in columns[2:]:
            # Unsupported or inaccessible host counters use -1. Keep them in
            # the per-rank table, but never average them into real usage.
            if key.startswith("cpu_") and prof_data[key] < 0:
                continue
            valid_counts[key] += 1
            min_values[key] = min(min_values[key], prof_data[key])
            max_values[key] = max(max_values[key], prof_data[key])
            sum_values[key] += prof_data[key]

    # Calculate average values
    avg_values = {}
    for key in columns[2:]:
        if valid_counts[key]:
            avg_values[key] = sum_values[key] / valid_counts[key]
        else:
            avg_values[key] = min_values[key] = max_values[key] = -1.0

    df = pd.DataFrame(data, columns=columns)
    summary_df = pd.DataFrame({"Avg": avg_values, "Max": max_values, "Min": min_values})

    if wandb.run:
        # Log the table
        table = wandb.Table(dataframe=df)
        _log_wandb_monitor_data({"DeviceMonitor/prof_data": table}, iteration)

        # Log summary statistics
        summary = {}
        for key in columns[2:]:
            summary[f"DeviceMonitor/min_{key}"] = min_values[key]
            summary[f"DeviceMonitor/max_{key}"] = max_values[key]
            summary[f"DeviceMonitor/avg_{key}"] = avg_values[key]

        _log_wandb_monitor_data(summary, iteration)
    return df, summary_df


class DeviceMonitor(EveryN):
    """
    A callback to monitor device (CPU/GPU) usage and log it at regular intervals.

    Args:
        every_n (int, optional): The frequency at which the callback is invoked. Defaults to 200.
        step_size (int, optional): The step size for the callback. Defaults to 1.
        save_s3 (bool, optional): Whether to save the monitoring data to S3. Defaults to False.
    """

    def __init__(
        self,
        every_n: int = 200,
        step_size: int = 1,
        save_s3: bool = False,
        upload_every_n_mul: int = 1,
        log_memory_detail: bool = True,
    ):
        super().__init__(every_n=every_n, step_size=step_size)
        self.name = self.__class__.__name__
        self.save_s3 = save_s3
        self.s3_save_fp = f"s3://rundir/{self.name}"
        self.upload_every_n = upload_every_n_mul * every_n

        self.log_memory_detail = log_memory_detail

    def on_train_start(self, model, iteration=0):
        torch.cuda.reset_peak_memory_stats()
        self.world_size = distributed.get_world_size()
        self.rank = distributed.get_rank()
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if self.rank == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"{self.name} callback: local_dir: {self.local_dir}")

        local_rank = int(os.getenv("LOCAL_RANK", 0))
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)

    def every_n_impl(
        self,
        trainer: ImaginaireTrainer,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        cur_process = psutil.Process(os.getpid())
        cpu_metrics = process_tree_memory(cur_process)
        # Allocator-reserved pinned RAM may grow while live allocations remain
        # stable. Older PyTorch versions do not expose these counters.
        pinned_stats = getattr(torch.cuda.memory, "host_memory_stats", lambda: {})()
        for label, key in (("allocated", "allocated_bytes.current"), ("reserved", "reserved_bytes.current")):
            value = pinned_stats.get(key)
            cpu_metrics[f"cpu_pinned_{label}_gb"] = value / (1024**3) if value is not None else -1.0

        peak_gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_gpu_mem_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        temp = torch.cuda.temperature()
        try:
            power = torch.cuda.power_draw()
        except Exception as e:
            log.warning(f"Failed to get power draw with error {e}")
            power = 0
        util = torch.cuda.utilization()
        clock = torch.cuda.clock_rate()

        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        nvml_used_gpu_mem_gb = memory_info.used / (1024**3)
        nvml_free_gpu_mem_gb = memory_info.free / (1024**3)

        prof_data = {
            **cpu_metrics,
            "peak_gpu_mem_gb": peak_gpu_mem_gb,
            "peak_gpu_mem_reserved_gb": peak_gpu_mem_reserved_gb,
            "nvml_used_gpu_mem_gb": nvml_used_gpu_mem_gb,
            "nvml_free_gpu_mem_gb": nvml_free_gpu_mem_gb,
            "temp": temp,
            "power": power,
            "util": util,
            "clock": clock,
        }

        data_list = [prof_data] * self.world_size
        # this is blocking by default
        if self.world_size > 1:
            torch.distributed.all_gather_object(data_list, prof_data)
            torch.distributed.barrier()

        df, summary_df = log_prof_data(data_list, iteration)
        if self.rank == 0:
            df.to_csv(os.path.join(self.local_dir, f"prof_data_{iteration:09d}.csv"), index=False)
            summary_df.to_csv(os.path.join(self.local_dir, f"summary_{iteration:09d}.csv"), index=True)
        if self.save_s3 and self.rank == 0:
            global_step = iteration // self.step_size
            should_run = global_step % self.upload_every_n == 0
            if should_run:
                easy_io.copyfile_from_local(
                    os.path.join(self.local_dir, f"prof_data_{iteration:09d}.csv"),
                    os.path.join(self.s3_save_fp, f"prof_data_{iteration:09d}.csv"),
                )
                easy_io.copyfile_from_local(
                    os.path.join(self.local_dir, f"summary_{iteration:09d}.csv"),
                    os.path.join(self.s3_save_fp, f"summary_{iteration:09d}.csv"),
                )
        if self.rank == 0:
            log.info(f"{self.name} Stats:\n{summary_df.to_string()}")
            if self.log_memory_detail:
                memory_stats = torch.cuda.memory_stats()
                if wandb.run:
                    wandb_memory_info = {f"mem/{key}": memory_stats[key] for key in memory_stats.keys()}
                    _log_wandb_monitor_data(wandb_memory_info, iteration)
                if self.save_s3:
                    global_step = iteration // self.step_size
                    should_run = global_step % self.upload_every_n == 0
                    if should_run:
                        easy_io.dump(
                            memory_stats,
                            os.path.join(self.s3_save_fp, f"memory_stats_{iteration:09d}.yaml"),
                        )

        torch.cuda.reset_peak_memory_stats()
