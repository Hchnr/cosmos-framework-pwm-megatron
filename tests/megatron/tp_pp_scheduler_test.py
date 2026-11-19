"""Source LR position survives reordered MCore groups and runtime restart."""

from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.training_backends.megatron_tp_pp.scheduler import NativeScheduler
from cosmos_framework.training_backends.megatron_tp_pp.source import CanonicalDCPReader
from cosmos_framework.utils.generator.optimizer import _lr_scheduler_cls


def optimizer(order):
    return torch.optim.SGD(
        [{"params": [torch.nn.Parameter(torch.ones(1))], "lr": value, "max_lr": value} for value in order]
    )


def test_native_lr_import_maps_by_identity_and_restarts(tmp_path):
    kwargs = dict(warm_up_steps=[3], cycle_lengths=[20], f_start=[0.0], f_max=[1.0], f_min=[0.1])
    source = optimizer([1e-3, 5e-3])
    native = torch.optim.lr_scheduler.LambdaLR(source, _lr_scheduler_cls("lambdacosine", **kwargs).schedule)
    for _ in range(11):
        source.step()
        native.step()
    dcp.save(native.state_dict(), checkpoint_id=tmp_path / "scheduler")
    target = optimizer([5e-3, 1e-3])
    scheduler = NativeScheduler(SimpleNamespace(optimizer=target), "lambdacosine", kwargs)
    assert scheduler.import_source(CanonicalDCPReader(tmp_path / "scheduler")) == native.last_epoch
    for _ in range(3):
        source.step()
        native.step()
        target.step()
        scheduler.step()
        assert [group["lr"] for group in target.param_groups] == list(reversed(native.get_last_lr()))
    saved = scheduler.state_dict()
    restarted_optimizer = optimizer([5e-3, 1e-3])
    restarted = NativeScheduler(SimpleNamespace(optimizer=restarted_optimizer), "lambdacosine", kwargs)
    restarted.load_state_dict(saved)
    target.step()
    scheduler.step()
    restarted_optimizer.step()
    restarted.step()
    assert [group["lr"] for group in target.param_groups] == [group["lr"] for group in restarted_optimizer.param_groups]
