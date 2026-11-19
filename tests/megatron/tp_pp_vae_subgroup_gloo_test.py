# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Exercise VAE checkpoint synchronization in nonzero-rank Gloo subgroups."""
import os
from pathlib import Path
import subprocess
import sys


def test_vae_sync_uses_subgroup_root():
    worker = Path(__file__).with_name('tp_pp_vae_subgroup_gloo_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', LD_LIBRARY_PATH='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    result = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=4', str(worker)],
        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout
    for rank in range(4):
        assert f'VAE_SUBGROUP_GLOO_PASS rank={rank}' in result.stdout, result.stdout
