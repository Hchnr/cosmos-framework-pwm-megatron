"""Stage-local depth partition of the simultaneous three-tower decoder."""

from megatron.core import parallel_state as ps
from megatron.core.enums import ModelType
from megatron.core.transformer.module import MegatronModule
from torch import nn

from cosmos_framework.training_backends.megatron_tp_pp.config import layer_range
from cosmos_framework.utils.nvtx import nvtx_range

from .layers import parallelize_layer


class PipelineStage(MegatronModule):
    """Build only this stage's layers, using global indices in checkpoint keys.

    ``layer_factory(global_index)`` should construct a meta layer for production.
    Preparation and output heads attach through first/last-stage adapters rather
    than putting training-specific dependencies in inference modules.
    """

    def __init__(
        self,
        config,
        layer_factory,
        *,
        layout,
        frozen_und=True,
        pre_process=None,
        post_process=None,
        input_factory=None,
        output_factory=None,
    ):
        super().__init__(config)
        self.model_type = ModelType.encoder_or_decoder
        self.share_embeddings_and_output_weights = False
        self.pre_process = ps.is_pipeline_first_stage()
        self.post_process = ps.is_pipeline_last_stage()
        self.layout = layout
        self.frozen_und = frozen_und
        if (
            pre_process is not None
            and input_factory is not None
            or post_process is not None
            and output_factory is not None
        ):
            raise ValueError("Provide either a module or a stage-local adapter factory")
        self.input_adapter = (
            (input_factory() if input_factory is not None else pre_process) if self.pre_process else None
        )
        self.output_adapter = (
            (output_factory() if output_factory is not None else post_process) if self.post_process else None
        )
        indices = layer_range(
            config.num_layers, config.pipeline_model_parallel_size, ps.get_pipeline_model_parallel_rank()
        )
        self.layers = nn.ModuleDict(
            {
                str(i): parallelize_layer(
                    layer_factory(i), config, ps.get_tensor_model_parallel_rank(), config.tensor_model_parallel_size
                )
                for i in indices
            }
        )
        self._input_tensor = None

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list) or len(input_tensor) != 1:
            raise ValueError("Three-tower stage expects one differentiable packed tensor")
        self._input_tensor = input_tensor[0]

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        from megatron.core.dist_checkpointing.mapping import ShardedTensor

        from cosmos_framework.training_backends.megatron_tp_pp.parameter_map import (
            canonical_local_name,
            stage_parameter_manifest,
        )

        if sharded_offsets:
            raise ValueError("PipelineStage already carries global layer identities")
        manifest = {canonical_local_name(k): v for k, v in stage_parameter_manifest(self).items()}
        state = self.state_dict(keep_vars=True)
        result = {}
        for name, value in state.items():
            if canonical_local_name(name) not in manifest:
                raise ValueError(f"Missing canonical buffer layout: {name}")
            entry = manifest[canonical_local_name(name)]
            offsets = (
                () if entry.axis is None else ((entry.axis, entry.tensor_rank, self.config.tensor_model_parallel_size),)
            )
            result[prefix + name] = ShardedTensor.from_rank_offsets(
                entry.canonical_name,
                value,
                *offsets,
                replica_id=(0, entry.tensor_rank if entry.axis is None else 0, ps.get_data_parallel_rank()),
            )
        return result

    def forward(self, batch):
        layout = batch.get("layout", self.layout)
        if layout != self.layout and not self.config.variable_seq_lengths:
            raise ValueError("Variable pipeline layouts require variable_seq_lengths=True")
        if self.pre_process:
            pack = self.input_adapter(batch) if self.input_adapter is not None else dict(batch["pack"])
            layout = batch.get("layout", layout)
        else:
            pack = dict(batch["metadata"])
            pack.update(layout.unpack(self._input_tensor, frozen_und=self.frozen_und))
            self._input_tensor = None
        if "publish_metadata" in batch:
            batch["publish_metadata"](batch)
        context = pack.pop("pipeline_context", None)
        for name, layer in self.layers.items():
            with nvtx_range(f"tp_pp.layer.{name}", required="module"):
                pack = layer(pack, None, None)[0]
        if context is not None:
            pack["pipeline_context"] = context
        if self.post_process and self.output_adapter is not None:
            return self.output_adapter(pack, batch)
        if batch.get("decoder_has_loss") is False:
            # Only the native dummy-head context reaches the loss. Detach the
            # decoder streams before the compiled backward boundary: AOT's
            # backward otherwise materializes None into zero cotangents.
            pack.update({key: pack[key].detach() for key in layout.keys if key != "pipeline_context"})
        return layout.pack(pack)
