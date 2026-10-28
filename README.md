<div align="center">

# PLSR: Progressive and Localized Super-Resolution<br>of 3D Objects via Localized Latent Voxel Diffusion

**ECCV 2026**

Yuxin Liu, Minshan Xie, Jiawen Liang, Runsong Zhu, Chi-Wing Fu, Tien-Tsin Wong

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2609.06436)
[![ECCV](https://img.shields.io/badge/ECCV-2026-blue.svg)](https://doi.org/10.1007/978-3-032-37359-5_16)
[![Models](https://img.shields.io/badge/Hugging%20Face-Models-yellow.svg)](https://huggingface.co/yxliu/PLSR)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

**PLSR** progressively refines the geometry and PBR materials of coarse 3D objects
using reference images. Built on TRELLIS.2, it applies localized latent diffusion
to overlapping 3D patches, enabling higher-resolution detail with bounded patch sizes.

## Installation

### Linux / WSL

Requirements: an NVIDIA GPU, CUDA Toolkit 12.8, a C++ compiler and Conda.
Run from the repository root:

```bash
. ./setup.sh --new-env --basic --flash-attn --pytorch3d --flexgemm \
  --diffoctreerast --nvdiffrast --cumesh --o-voxel --mapanything --nvdiffrec
```

`--new-env` creates and activates `plsr` with Python 3.12 and PyTorch 2.9.1 (CUDA 12.8).
To install into an existing compatible environment, activate it and omit `--new-env`.
Set `CUDA_HOME` if your CUDA toolkit is not at the default location.

Dependency groups can be installed together or separately:

| Option | Component |
| --- | --- |
| `--basic` | Python dependencies and build tools |
| `--flash-attn` | FlashAttention |
| `--pytorch3d` / `--flexgemm` | Local attention and sparse operations |
| `--diffoctreerast` / `--nvdiffrast` | Patch visibility and texture baking |
| `--cumesh` / `--o-voxel` | Mesh processing and voxel conversion |
| `--mapanything` / `--nvdiffrec` | Camera estimation and anchor rendering |

For PyTorch SDPA, omit `--flash-attn` and set `ATTN_BACKEND=sdpa` and
`SPARSE_ATTN_BACKEND=sdpa`. See `bash setup.sh --help` for all options.
Required model weights are downloaded automatically on first run.

### Windows

With Conda and Visual Studio 2022 C++ Build Tools installed, run the following in
PowerShell. The Windows installer uses SDPA; it was adapted with AI and has only
been tested in the author's own environment.

```powershell
.\setup.ps1 -NewEnv -Basic -PyTorch3D -FlexGemm -DiffOctreeRast -NvDiffRast -CuMesh -OVoxel -MapAnything -NvDiffRec
```

## Usage

Activate the environment with `conda activate plsr`. Three input modes are supported:

**1. Image only — generate a coarse object, then refine it.**

```bash
python -m plsr infer --image assets/images/00000.jpg \
  --output-dir outputs/00000 --resolutions 640 1280 --low-memory
```

**2. Coarse object and images — estimate cameras automatically.**

```bash
python -m plsr infer --mesh /path/to/coarse.glb \
  --image /path/to/front.png --image /path/to/side.png \
  --output-dir outputs/mesh --resolutions 640 1280 --low-memory --independent-camera-estimation
```

One image also works. **Multiple images currently require an input object.**
RGB inputs are automatically processed to remove the background; images with
useful alpha retain their foreground.

**3. Coarse object, images and known cameras — use a JSON configuration.**

```bash
python -m plsr infer assets/venice_mask/known_cameras.json --low-memory
```

The [Venice mask example](assets/venice_mask/README.md) includes a coarse GLB,
six reference images and camera matrices. Copy its configuration for your own data;
paths are relative to the JSON file. Cameras use 3×3 intrinsics and 4×4 OpenCV
world-to-camera extrinsics in the input mesh's coordinate system. Intrinsics are
normalized by image width/height, or given in pixels with `"intrinsics_units": "pixels"`.
The accompanying [estimated_cameras.json](assets/venice_mask/estimated_cameras.json)
runs the same example without supplied cameras.

### Gradio

For an interactive interface with image/model previews and alignment diagnostics:

```bash
pip install -r requirements-gradio.txt
python app.py
```

Open `http://127.0.0.1:7860`. Upload one image with or without a coarse GLB, or
multiple images with a GLB. Cameras are estimated automatically. Coarse models,
alignment overlays and refined models appear as their stages complete; results
are saved under `outputs/gradio` and can be downloaded from the interface.

### Outputs and useful options

The output directory contains `mesh_<resolution>.glb`, saved `latents_<resolution>.npz`,
`cameras.json` and `report.json`. Silhouette overlays in `view_diagnostics/` show
image-only regions in cyan, mesh-only regions in orange and overlap in light gray;
`view_diagnostics.json` records each view's IoU. Views below the threshold are
excluded, and inference stops if none remain. Poor silhouette overlap may reduce
the quality of the refined output.

| Option | Purpose |
| --- | --- |
| `--resolutions 640 1280 2560` | Progressive refinement levels |
| `--low-memory` | Smaller batches, camera CPU offload and coarse/refinement decode chunks capped at 8000 latent cells |
| `--steps`, `--shape-guidance`, `--pbr-guidance`, `--seed` | Sampling controls; omitted values use model defaults |
| `--min-view-iou 0.6` | Minimum silhouette IoU for retaining a view as a conditioning input |
| `--independent-camera-estimation` | Recommended for multi-image inputs on low-VRAM GPUs; may occasionally improve camera estimation. Joint estimation is the default |
| `--decode-cell-limit 8000` | Override the maximum latent cells per decode chunk without changing resolution |
| `--texture-size 4096`, `--decimation-target 500000` | Texture size and face cap per exported chunk |
| `--texture-budget 8192` | Optional total pixel budget (8192² per material map) for coarse and refinement exports; default `0` disables it. Smaller textures may lose fine detail |
| `--decode-intermediate` | Export intermediate refinement levels |

Rerun the same command to resume from saved results; use a new output directory
when changing inputs or settings. See `python -m plsr infer --help` for all options.

## Acknowledgments

Built on [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and O-Voxel, with
DINOv3, MapAnything and BiRefNet for image features, camera estimation and background removal.

## Citation

```bibtex
@inproceedings{liu2026plsr,
  title={{PLSR}: Progressive and Localized Super-Resolution of {3D} Objects via Localized Latent Voxel Diffusion},
  author={Liu, Yuxin and Xie, Minshan and Liang, Jiawen and Zhu, Runsong and Fu, Chi-Wing and Wong, Tien-Tsin},
  booktitle={Computer Vision -- ECCV 2026},
  pages={281--299},
  year={2026},
  publisher={Springer},
  doi={10.1007/978-3-032-37359-5_16}
}
```
