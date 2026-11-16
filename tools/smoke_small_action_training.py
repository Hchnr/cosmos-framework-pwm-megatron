"""Run the real training entry for preflight without saving any smoke checkpoints."""

import runpy

from cosmos_framework.checkpoint.dcp import DistributedCheckpointer

DistributedCheckpointer.save = lambda *args, **kwargs: None
runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
