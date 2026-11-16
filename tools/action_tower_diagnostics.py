"""Standalone evaluation-only probes; never imported by production model code."""

import torch


def full_gen_diagnostic(video_module, action_module, x, pack):
    if torch.is_grad_enabled() or video_module.training or action_module.training:
        raise RuntimeError("Full-GEN diagnostic requires eval mode and disabled gradients; training is forbidden")
    vi, ai = pack["_video_indices"], pack["_action_indices"]
    video, action = video_module(x), action_module(x)
    return torch.zeros_like(video).index_copy(0, vi, video[vi]).index_copy(0, ai, action[ai])


def install_full_gen_diagnostic():
    from cosmos_framework.model.generator.mot import action_tower

    print("DIAGNOSTIC ONLY: redundant full-GEN evaluation enabled; results are not production parity", flush=True)
    action_tower.routed_gen = full_gen_diagnostic
