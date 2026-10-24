# Inference dependencies by input mode

The base PLSR path starts from a supplied coarse mesh, a reference image with
useful alpha, and complete camera intrinsics and extrinsics. Single-image coarse
generation uses the same compiled dependencies, but also needs the original
TRELLIS.2 weights and its sparse-structure decoder. Opaque/RGB references need
RMBG-2.0 background-removal weights. Missing camera fields enable the separate
MapAnything and rendered-anchor path.

| Native package | Why inference uses it | Required when |
| --- | --- | --- |
| PyTorch3D | KNN neighbors for PLSR's local attention | Every refinement run |
| FlexGEMM and Triton | Sparse convolution and voxel attribute sampling | Every refinement run |
| diffoctreerast | Renders patch visibility masks for image conditioning | Every refinement run |
| O-Voxel | Encodes the input mesh and decodes Shape/PBR latents | Every refinement run |
| CuMesh | Mesh representation, remeshing, and GLB baking | Every refinement run |
| nvdiffrast | Texture baking and material sampling | Every refinement run |
| nvdiffrec renderutils | Lit environment maps for camera-estimation anchors | Only when camera fields are missing |

The octree renderer contains an old placeholder image path when
diffoctreerast is absent. That placeholder is not a valid replacement for
PLSR's patch visibility masks, so diffoctreerast stays in the base install.

PyTorch SDPA is the tested Windows attention backend and needs no separate
attention extension. The tested Linux configuration uses FlashAttention.
xFormers is an alternative backend; neither xFormers nor FlashAttention is
needed when both attention backends are set to `sdpa`. `spconv` and
`torchsparse` are alternative sparse-convolution backends and are not used by
the published FlexGEMM configuration. Installing them does not replace
FlexGEMM's other inference uses.

The direct Python requirements are grouped by their actual call sites:

| Packages | Use |
| --- | --- |
| PyTorch, TorchVision, NumPy, Pillow | Model execution, DINOv3 preprocessing, images, and mesh data |
| Transformers, safetensors, PEFT | Local DINOv3/RMBG models, weight files, and PLSR LoRA layers |
| Hugging Face Hub | Downloads pipeline configuration and model files |
| trimesh, OpenCV, SciPy, Kornia, easydict | Mesh input/export, image and patch processing, and octree visibility rendering |
| tqdm | Sampling and mesh postprocessing progress |
| plyfile, zstandard | Imported by O-Voxel's I/O package when O-Voxel loads |
| timm | Imported by the local RMBG-2.0 model code for opaque/RGB images |

`timm` remains in the base requirements because opaque/RGB input is common;
useful-alpha RGBA input skips the background-removal model and its weights.
TorchVision is installed with PyTorch before the root `requirements.txt`.
PEFT installs Accelerate and psutil through its own package metadata; they are
not listed or versioned separately by the installers. SciPy and `easydict` are loaded by the octree visibility
renderer, and OpenCV also loads through mesh postprocessing.

Missing cameras additionally need utils3d, imageio, MapAnything with its regular
Python dependencies (including UniCeption and torchaudio), nvdiffrec renderutils,
and the bundled HDR environment map. These are selected by installer options.
Complete known cameras skip that entire branch. On Windows, omit `-MapAnything` and `-NvDiffRec`; on Linux, omit
`--mapanything` and `--nvdiffrec` from the installation command.

`imageio` is needed by nvdiffrec's imported image utilities and is included only
in the camera installation options. `imageio-ffmpeg` and the original demo/training
dependencies are not needed for this inference entry point.
PLSR, TRELLIS.2, DINOv3, RMBG-2.0 and MapAnything weights are loaded separately
from their HF repositories or downloaded HF directories; the
[inference guide](INFERENCE_USAGE.md) describes the model source overrides.
The DINOv3 encoder uses Transformers and does not require a separate `dinov3`
Python package.

The root `requirements.txt` contains basic Python dependencies and the
recommended Transformers/PEFT versions. The installer selects Torch, attention
and native packages. Ordinary Python tools and transitive dependencies follow
their upstream package requirements.
