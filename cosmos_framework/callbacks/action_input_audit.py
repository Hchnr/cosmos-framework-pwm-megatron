"""Optional input identities for online/offline action comparisons."""

import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.utils.callback import Callback


class ActionInputAudit(Callback):
    def __init__(self, seed_after_init: int | None = None):
        super().__init__()
        self.seed_after_init = seed_after_init

    def on_train_start(self, model, iteration=0):
        self.path = Path(self.config.job.path_local) / "input_audit"
        self.path.mkdir(parents=True, exist_ok=True)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.seed_after_init is not None:
            from cosmos_framework.utils import misc

            misc.set_random_seed(seed=self.seed_after_init, by_rank=True)

    def on_training_step_batch_start(self, model, data_batch, iteration=0):
        digest = hashlib.sha256()

        def update(value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().contiguous()
                digest.update(str((value.dtype, tuple(value.shape))).encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            elif isinstance(value, (list, tuple)):
                for child in value:
                    update(child)
            else:
                digest.update(repr(value).encode())

        for key in ("source_sample_index", "text_token_ids", "action", "domain_id", "sequence_plan"):
            update(data_batch.get(key))
        with (self.path / f"rank_{self.rank}.jsonl").open("a") as f:
            f.write(json.dumps(dict(iteration=iteration, signature=digest.hexdigest())) + "\n")
