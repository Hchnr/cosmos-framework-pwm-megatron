# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Run the nested-checkpoint chunk candidate through the TP/native tiny oracle."""

import json
import os
import sys
from pathlib import Path

from tools import run_tp_pp_checkpointed_chunked_mlp_math as candidate
from tools import run_tp_pp_chunked_mlp_math as previous


def main():
    counts = {"chunk_checkpoints": 0}
    checkpoint = candidate.checkpoint

    def observed(*args, **kwargs):
        counts["chunk_checkpoints"] += 1
        return checkpoint(*args, **kwargs)

    candidate.checkpoint = observed
    previous.install = candidate.install
    from tools import qualify_tp_pp_flat_checkpoint_math as qualifier

    qualifier.main()
    if counts["chunk_checkpoints"] == 0:
        raise RuntimeError("Nested chunk checkpoints were not exercised")
    # The existing qualifier writes the detailed native/TP reports and includes
    # a required --output; keep this independent witness beside them.
    output_index = sys.argv.index("--output") + 1
    output = Path(sys.argv[output_index])
    target = output.with_name(f"{output.stem}.checkpointed_chunk.rank{os.environ['RANK']}.json")
    with target.open("x") as handle:
        json.dump({"passed": True, **counts, "full_model_acceptance": False}, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
