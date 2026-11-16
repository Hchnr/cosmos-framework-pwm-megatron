# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Training callback that defers AOT compilation of the VAE tokenizer.

The actual compilation logic lives in
:meth:`~cosmos_framework.model.generator.tokenizers.wan2pt2_vae_4x16x16.Wan2pt2VAEInterface.compile_encode`.
This module provides a :class:`CompileTokenizer` callback that invokes it
at the right point during training (at optimizer iteration
``compile_after_iterations``, to avoid NCCL timeouts during CUDA/cuDNN
warm-up).  The optimizer iteration is shared across ranks; counting local
microbatch-hook calls is unsafe with context-parallel data windows.

Typical config usage
--------------------
.. code-block:: python

    CompileTokenizer(
        enabled=True,
        compile_after_iterations=3,
        warmup_resolutions=["256", "480", "720"],
    )
"""

from collections.abc import Sequence
from typing import Literal

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel
from cosmos_framework.utils import log
from cosmos_framework.utils.callback import Callback
from cosmos_framework.utils.generator.rand_state import preserve_training_rng


class CompileTokenizer(Callback):
    """Training callback that defers AOT compilation of the VAE tokenizer.

    Hooks into ``on_training_step_start``.  At optimizer iteration
    ``compile_after_iterations`` it calls
    ``Wan2pt2VAEInterface.compile_encode`` to compile and load all chunk
    variants.  A local latch makes repeated microbatch hooks in that iteration
    and every subsequent step no-ops.
    """

    def __init__(
        self,
        enabled: bool = False,
        compile_after_iterations: int = 3,
        warmup_resolutions: Sequence[str] | None = None,
        aspect_ratio: str | None = None,
        backend: Literal["cudagraphs", "inductor"] = "inductor",
        mode: Literal["reduce-overhead", "max-autotune"] | None = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        spatial_sizes: Sequence[Sequence[int]] | None = None,
        require_full_coverage: bool = False,
    ):
        """
        Args:
            enabled: Master switch.  When ``False`` the callback is a
                complete no-op and no compilation occurs.
            compile_after_iterations: How many training steps to skip
                before triggering compilation.  The default (3) lets CUDA
                context setup and Transformer compilation finish first.
            warmup_resolutions: Resolution keys (e.g. ``["256", "480", "720"]``)
                to AOT-compile.  Should include every resolution used in
                training. Required unless explicit ``spatial_sizes`` are supplied.
            spatial_sizes: Wan pixel (H, W) canvases, replacing standard buckets.
            require_full_coverage: Wan fails on incomplete AOT coverage instead of falling back.
            aspect_ratio: Optional single aspect-ratio bucket (for example
                ``"1,1"``). When omitted, all aspect ratios for every warmup
                resolution are compiled.
        """
        super().__init__()
        self.enabled: bool = enabled
        self.compile_after_iterations: int = compile_after_iterations
        self._has_compiled: bool = False
        # Hydra can coerce numeric-looking CLI values such as ``"256"`` to
        # integers after launcher shell parsing. Resolution lookup tables use
        # string keys, so normalize them at the callback boundary.
        self.warmup_resolutions: Sequence[str] | None = (
            None if warmup_resolutions is None else [str(resolution) for resolution in warmup_resolutions]
        )
        self.spatial_sizes = spatial_sizes
        self.require_full_coverage = require_full_coverage
        self.aspect_ratio = aspect_ratio
        self.backend: Literal["cudagraphs", "inductor"] = backend
        self.mode: Literal["reduce-overhead", "max-autotune"] | None = mode
        self.fullgraph: bool = fullgraph
        self.dynamic: bool = dynamic

        if self.enabled:
            if self.compile_after_iterations < 0:
                raise ValueError("compile_after_iterations must be non-negative")
            if self.spatial_sizes is not None:
                if not self.spatial_sizes or self.aspect_ratio is not None:
                    raise ValueError("spatial_sizes must be non-empty and cannot be combined with aspect_ratio")
                self.warmup_resolutions = self.warmup_resolutions or []
            elif self.warmup_resolutions is None:
                raise ValueError("warmup_resolutions must be provided when enabled, got None")
            if not self.warmup_resolutions and self.spatial_sizes is None:
                raise ValueError("warmup_resolutions must be a non-empty list when enabled, got an empty list")

    def on_training_step_start(
        self, model: OmniMoTModel, data_batch: dict[str, torch.Tensor], iteration: int = 0
    ) -> None:
        """Called at the start of every training step.

        On the first call whose optimizer iteration is at least
        ``compile_after_iterations``, triggers AOT compilation via
        ``tokenizer.compile_encode``.

        Args:
            model: The OmniMoTModel whose ``tokenizer_vision_gen`` will be compiled.
            data_batch: Current training batch (unused, required by Callback API).
            iteration: Current optimizer iteration.  This is used instead of a
                local hook-call counter so every distributed rank reaches AOT at
                the same training boundary even with gradient accumulation or CP.
        """
        if not self.enabled:
            return

        tokenizer = model.tokenizer_vision_gen
        if tokenizer is None:
            log.info("CompileTokenizer disabled because the vision tokenizer is not loaded", rank0_only=False)
            self.enabled = False
            return

        if isinstance(tokenizer, torch.jit.ScriptModule):
            log.critical(
                f"The Tokenizer model {type(tokenizer)} is a JIT model, "
                "which is not compilable. The Tokenizer will not be compiled.",
                rank0_only=False,
            )
            self.enabled = False
            return

        if not self._has_compiled and iteration >= self.compile_after_iterations:
            # Latch before entering the distributed compiler.  A raised exception
            # terminates this run; repeated microbatch hooks must never start a
            # second global AOT sequence.
            self._has_compiled = True
            if self.warmup_resolutions is not None:
                log.info(
                    f"Starting tokenizer AOT at optimizer iteration {iteration}",
                    rank0_only=False,
                )
                extra = {}
                if self.spatial_sizes is not None:
                    extra["spatial_sizes"] = self.spatial_sizes
                if self.require_full_coverage:
                    extra["require_full_coverage"] = True
                with preserve_training_rng():
                    tokenizer.compile_encode(
                        self.warmup_resolutions,
                        output_dir=self.config.job.path_local,
                        aspect_ratio=self.aspect_ratio,
                        backend=self.backend,
                        mode=self.mode,
                        fullgraph=self.fullgraph,
                        dynamic=self.dynamic,
                        **extra,
                    )
