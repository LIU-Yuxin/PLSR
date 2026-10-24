# Installing PLSR inference on Linux

The installer follows [TRELLIS.2's installation workflow](https://github.com/microsoft/TRELLIS.2#️-installation):
create an environment with `--new-env`, install Python dependencies with
`--basic`, and select native packages individually. Options can be combined.
For native Windows, see [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md).

## Prerequisites

Install Conda, Git, an NVIDIA driver, a C++ compiler (GCC/G++), and CUDA Toolkit
12.8 before building CUDA extensions. The PyTorch wheel supplies runtime
libraries; it does not supply the complete compiler toolkit.
If multiple toolkits are installed, select the matching one, for example:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

The extension build tools normally target the visible GPU. Set
`TORCH_CUDA_ARCH_LIST` only when building for a different target GPU. Set
`MAX_JOBS` if you need to limit compilation memory use.

## Install

From the repository root, install the complete single-image and mesh workflow:

```bash
. ./setup.sh --new-env --basic --flash-attn --pytorch3d --flexgemm \
  --diffoctreerast --nvdiffrast --cumesh --o-voxel --mapanything --nvdiffrec
```

`--new-env` creates a Conda environment named `plsr` with Python 3.12 and installs
PyTorch/torchvision for CUDA 12.8. Sourcing the script with `. ./setup.sh` keeps
that environment active in your current shell.

For an existing environment with the recommended PyTorch stack, omit
`--new-env` and run:

```bash
conda activate plsr
bash setup.sh --basic --flash-attn --pytorch3d --flexgemm \
  --diffoctreerast --nvdiffrast --cumesh --o-voxel --mapanything --nvdiffrec
```

For a supplied mesh with complete camera intrinsics and extrinsics, omit
`--mapanything` and `--nvdiffrec`. See
[the dependency guide](INFERENCE_DEPENDENCIES.md) for the input modes.

The script installs selected groups in dependency order and stops on a failed
command. After fixing the cause, rerun just the failed package, for example:

```bash
bash setup.sh --cumesh
bash setup.sh --o-voxel
```

Run `bash setup.sh --help` for all options. `--mapanything` installs its camera
Python dependencies and matching torchaudio; `--nvdiffrec` installs the rendered
camera-anchor extension. Both are needed when camera fields are missing.

FlashAttention can be omitted by setting both backends to SDPA before inference:

```bash
export ATTN_BACKEND=sdpa
export SPARSE_ATTN_BACKEND=sdpa
```

Set these variables again in each new shell. FlexGEMM still uses Triton, which
PyTorch installs automatically on Linux.

## Recommended core versions

This combination was used for Linux/WSL model loading and inference:

| Component | Recommended version |
| --- | --- |
| Python / CUDA Toolkit | 3.12 / 12.8 |
| PyTorch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| torchaudio (camera estimation) | 2.9.1+cu128 |
| Triton / FlashAttention | 3.5.1 / 2.8.3 |
| Transformers / PEFT | 4.57.3 / 0.18.0 |

The root `requirements.txt` lists basic Python dependencies, with recommended
versions for Transformers and PEFT. `setup.sh` selects PyTorch, torchvision,
matching torchaudio and FlashAttention versions. PyTorch installs its matching
Triton version automatically. Ordinary image/mesh tools have no PLSR-specific pins.
Accelerate, psutil and other transitive dependencies are installed by their
parent packages. Upstream packages may have their own version requirements.

The installers pin these native source revisions:

| Package | Source revision |
| --- | --- |
| PyTorch3D | `75ebeeaea0908c5527e7b1e305fbc7681382db47` |
| FlexGEMM | `1d83f0f6e457ddc8d948218ee79fb68808315688` |
| diffoctreerast | `b09c20b84ec3aace4729e6e18a613112320eca3a` |
| nvdiffrast | `253ac4fcea7de5f396371124af597e6cc957bfae` |
| CuMesh | `12289e1062f0603f2f0d0771b02e1395d247f26f` |
| MapAnything | `f065d13383ae84a7f80f0e0104a4d12afc7c7c82` |
| nvdiffrec renderutils | `b296927cc7fd01c2ac1087c8065c4d7248f72da4` |
| O-Voxel | Bundled `o-voxel/` source |

DiffOctreeRast uses the public `JeffreyXiang/diffoctreerast` repository.
The camera dependency group also selects UniCeption 0.1.6, matching the
MapAnything adapter used for optional camera batching.

Pip fetches Git submodules for the VCS installations. O-Voxel is installed with
`--no-deps` to retain the selected CuMesh/FlexGEMM revisions. Its Eigen headers
and license notices are included in the inference release.

## Model access

Package installation does not download model weights. PLSR defaults to
`yxliu/PLSR`, TRELLIS.2 to `microsoft/TRELLIS.2-4B`, and DINOv3 to
`facebook/dinov3-vitl16-pretrain-lvd1689m`. Obtain DINOv3 access and authenticate
with Hugging Face before the first download; see
[Meta's model instructions](https://github.com/facebookresearch/dinov3#pretrained-models).

`PLSR_REPO_ID`, `TRELLIS2_REPO_ID`, `DINO_REPO_ID`, `MAPANYTHING_REPO_ID` and
`RMBG_REPO_ID` accept HF repo IDs or downloaded HF directories. Python API
arguments take precedence over these environment variables. See
[MODEL_REPO.md](MODEL_REPO.md) and [INFERENCE_USAGE.md](INFERENCE_USAGE.md) for
loading examples. HF manages downloading and caching; `HF_HOME` changes the
cache location.

## Check and run

Run from the repository root:

```bash
python scripts/check_inference_environment.py --camera
python scripts/check_cuda_extensions.py --camera
python -m plsr infer --image /path/to/reference.png --output-dir /path/to/result
```

Omit `--camera` from the checks when using a supplied mesh and complete cameras.
The first check reads package metadata; the second executes small GPU operations
and imports native extensions. They are separate from model loading and inference.

## FlashAttention build cache

If FlashAttention reports `Invalid cross-device link` while packaging a downloaded
wheel, keep the pip cache and temporary build directory on the same filesystem:

```bash
PLSR_INSTALL_TMP="$(mktemp -d)"
TMPDIR="$PLSR_INSTALL_TMP" PIP_CACHE_DIR="$PLSR_INSTALL_TMP/pip-cache" \
  bash setup.sh --flash-attn
```

This only changes cache placement for that command. Existing
`TORCH_CUDA_ARCH_LIST` and `FLASH_ATTN_CUDA_ARCHS` settings are retained;
the installer does not reduce architecture targets to the current GPU.

For a fresh-machine verification, see [VALIDATION.md](VALIDATION.md).
