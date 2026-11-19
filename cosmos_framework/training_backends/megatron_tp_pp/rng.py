"""Native rank training RNG snapshots, independent of pipeline execution order."""

from contextlib import contextmanager

import torch

from cosmos_framework.utils.generator.rand_state import (
    get_rand_state_dict,
    preserve_training_rng,
    set_rand_state_dict,
)

RNG_FIELDS = (
    "torch",
    "torch_cuda",
    "numpy_packed_len",
    "numpy_packed_bytes",
    "random_packed_len",
    "random_packed_bytes",
)


class LogicalTrainingRNG:
    def __init__(self, states):
        self.states = states
        self._active = False
        for rank, state in states.items():
            if set(state) != set(RNG_FIELDS) or any(value.device.type != "cpu" for value in state.values()):
                raise ValueError(f"Incomplete CPU-stored logical RNG snapshot: rank {rank}")

    @classmethod
    def from_source(cls, reader, logical_ranks):
        values = reader.read_many(f"rng_state_{rank}.{key}" for rank in logical_ranks for key in RNG_FIELDS)
        return cls({rank: {key: values[f"rng_state_{rank}.{key}"] for key in RNG_FIELDS} for rank in logical_ranks})

    @contextmanager
    def activate(self, rank):
        if self._active:
            raise RuntimeError("Logical RNG activation cannot be nested")
        with preserve_training_rng():
            self._active = True
            try:
                set_rand_state_dict(self.states[rank])
                yield
            finally:
                self.states[rank] = get_rand_state_dict()
                self._active = False

    def draw_loader_base_seeds(self, rank, sources):
        """Match a fresh native SharedPrefetchDataLoader._start after extend."""
        with self.activate(rank):
            return [int(torch.empty((), dtype=torch.int64).random_().item()) for _ in range(sources)]

    def state_dict(self):
        if self._active:
            raise RuntimeError("Checkpoint logical RNG outside active preparation")
        return {rank: {name: value.clone() for name, value in state.items()} for rank, state in self.states.items()}
