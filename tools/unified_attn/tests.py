# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Select a local GPU before importing pytest suites under torchrun."""

import os
import sys

import pytest
import torch

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
sys.exit(pytest.main(["-c", "/dev/null", "--noconftest", "-p", "no:env", "-q", *sys.argv[1:]]))
