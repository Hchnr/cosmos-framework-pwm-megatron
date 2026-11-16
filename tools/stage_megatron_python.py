"""Stage Python sources locally; retain the reference CUDA binaries by symlink."""

import argparse
import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-site", type=Path, required=True)
    parser.add_argument("--target-site", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("/opt/cosmos-megatron-runtime"))
    args = parser.parse_args()
    source = args.source_site.resolve()
    fingerprint = str(source) + str(source.stat().st_mtime_ns) + (source / "torch/version.py").read_text()
    target = args.cache_root / hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
    files = []
    packages = (
        "torch",
        "torchvision",
        "transformers",
        "onnxscript",
        "onnx_ir",
        "onnx",
        "sympy",
        "numpy",
        "scipy",
        "einops",
        "jinja2",
        "mpmath",
        "networkx",
        "filelock",
        "timm",
    )
    for package in packages:
        root = source / package
        if not root.is_dir():
            continue
        # Many wheels use $ORIGIN/../../<package>.libs for bundled BLAS and
        # other native dependencies. Preserve those sibling library paths.
        sibling = source / f"{package}.libs"
        if sibling.exists() and not (target / sibling.name).exists():
            target.mkdir(parents=True, exist_ok=True)
            (target / sibling.name).symlink_to(sibling, target_is_directory=True)
        for directory, dirs, names in os.walk(root):
            directory = Path(directory)
            output = target / directory.relative_to(source)
            output.mkdir(parents=True, exist_ok=True)
            for name in list(dirs):
                item = directory / name
                if name in ("__pycache__", "include", "lib", "share", "test", "tests") or item.is_symlink():
                    dirs.remove(name)
                    if name != "__pycache__" and not (output / name).exists():
                        (output / name).symlink_to(item, target_is_directory=True)
            for name in names:
                item, destination = directory / name, output / name
                if destination.exists():
                    continue
                if ".so" in name or name.endswith(".a"):
                    destination.symlink_to(item)
                else:
                    files.append((item, destination))
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda pair: shutil.copy2(*pair), files))
    (args.target_site / "00-node-local-runtime.pth").write_text(str(target) + "\n")
    print(f"Node-local Python runtime: {target}; staged {len(files)} files", flush=True)


if __name__ == "__main__":
    main()
