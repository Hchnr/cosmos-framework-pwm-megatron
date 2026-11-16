# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Diagnostic-only determinism with compile kept enabled, unlike train --deterministic.

The ordinary attention frontend defaults deterministic=False; setting a random
seed alone does not change FA3's backward accumulation algorithm.
"""

import functools
import importlib
import runpy

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
module = importlib.import_module("cosmos_framework.model.generator.mot.attention")
module.attention = functools.partial(module.attention, deterministic=True)
runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
