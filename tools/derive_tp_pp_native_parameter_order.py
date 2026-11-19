"""Derive native optimizer parameter order from a meta model and canonical metadata."""

import argparse
import json
from pathlib import Path

import torch


def native_parameter_order(model_config, manifest):
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import Cosmos3VFMNetwork
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel, _prepare_full_model_instance
    from cosmos_framework.utils.lazy_config import instantiate

    instance, _, _ = _prepare_full_model_instance(
        model_config.vlm_config.model_instance, load_vlm_visual=False, include_generation_tower=True
    )
    language = instantiate(instance["config"])
    view = OmniMoTModel.__new__(OmniMoTModel)
    view.__dict__.update(config=model_config, tokenizer_vision_gen=None)
    network = view.build_network_config(language)
    with torch.device("meta"):
        model = Cosmos3VFMNetwork(instantiate(instance), network)
    parameters = dict(model.named_parameters())
    names = [
        name
        for name in model.state_dict()
        if name in parameters and manifest["parameters"]["net." + name]["optimizer_fields"]
    ]
    expected = {
        name.removeprefix("net.")
        for name, entry in manifest["parameters"].items()
        if entry["optimizer_fields"]
    }
    if set(names) != expected:
        raise ValueError("Native meta-model optimizer parameter coverage differs from canonical metadata")
    return names


def main():
    from cosmos_framework.configs.toml_config.sft_config import load_experiment_from_toml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toml", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_experiment_from_toml(str(args.toml))
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    names = native_parameter_order(config.model.config, manifest)
    result = dict(
        names=names,
        source="native meta model state_dict registration order and canonical optimizer_fields metadata",
        reads_reference_values=False,
        reads_parameter_payloads=False,
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
