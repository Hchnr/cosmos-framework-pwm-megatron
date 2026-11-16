# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Merge a slim Understanding training DCP with a previous full Generation DCP."""

from cosmos_framework.inference.common.init import init_script

init_script(env={"COSMOS_DEVICE": "cpu", "COSMOS_TRAINING": "1"})

import json
from pathlib import Path
from typing import Annotated

import pydantic
import tyro

from cosmos_framework.scripts._merge_understanding_dcp import merge_understanding_dcp


class Args(pydantic.BaseModel):
    generation_checkpoint_path: Path
    """Previous full DCP that supplies Generation/Action weights and model config."""
    understanding_checkpoint_path: Path
    """Slim DCP produced by pure Understanding training."""
    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """New full model-only DCP directory; it must not already exist."""


def main() -> None:
    args = tyro.cli(Args, description=__doc__, config=(tyro.conf.OmitArgPrefixes,))
    report = merge_understanding_dcp(
        args.generation_checkpoint_path,
        args.understanding_checkpoint_path,
        args.output_path,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
