"""Inspect inference dependency metadata without importing Torch or using CUDA."""

import argparse
from importlib import metadata
import json
import os
from pathlib import Path


def check_environment(camera=False):
    root = Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text().splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    names = [line.split("==", 1)[0] for line in requirements]
    dense_attention = os.environ.get("ATTN_BACKEND", "flash_attn")
    if dense_attention not in {"flash_attn", "flash_attn_3", "xformers", "sdpa", "naive"}:
        dense_attention = "flash_attn"
    sparse_attention = os.environ.get("SPARSE_ATTN_BACKEND", dense_attention)
    if sparse_attention not in {"flash_attn", "flash_attn_3", "xformers", "sdpa"}:
        sparse_attention = "flash_attn"
    attention_packages = set()
    for backend in (dense_attention, sparse_attention):
        if backend.startswith("flash_attn"):
            attention_packages.add("flash-attn")
        elif backend == "xformers":
            attention_packages.add("xformers")
    names += ["torch", "torchvision", *sorted(attention_packages), "pytorch3d", "flex-gemm",
              "diffoctreerast", "nvdiffrast", "cumesh", "o-voxel"]
    if camera:
        names += ["utils3d", "mapanything", "uniception", "torchaudio", "nvdiffrec-render", "imageio"]
    constraints = dict(line.split("==", 1) for line in requirements if "==" in line)
    packages, missing, differences = {}, [], {}
    for name in names:
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        packages[name] = installed
        expected = constraints.get(name)
        if expected and installed.split("+", 1)[0] != expected:
            differences[name] = {"installed": installed, "tested": expected}
    return {"status": "missing_packages" if missing else "packages_present",
            "attention_backends": {"dense": dense_attention, "sparse": sparse_attention},
            "camera_dependencies": camera, "packages": packages, "missing": missing,
            "version_differences": differences, "cuda_execution_checked": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="Include optional camera estimation dependencies")
    report = check_environment(parser.parse_args().camera)
    print(json.dumps(report, indent=2))
    return int(bool(report["missing"]))


if __name__ == "__main__":
    raise SystemExit(main())
