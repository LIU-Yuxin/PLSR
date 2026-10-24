# Native Windows inference installation

The Windows installer was automatically adapted by AI. Windows support has
only been tested in the author's own environment; review `setup.ps1` before
using it. Before installing anything, the script displays this warning and
asks you to type `y`. Any other response cancels installation.

Install Miniconda, Git, an NVIDIA driver, and Visual Studio 2022 Build Tools
with **Desktop development with C++** and a Windows SDK. Clone PLSR to a
**Windows drive**, rather than a WSL `\\wsl.localhost` path.
The recommended core combination is Python 3.12, PyTorch 2.9.1 cu128,
CUDA Toolkit 12.8, Triton Windows 3.5.1.post24, and PyTorch SDPA.
The script installs CUDA Toolkit inside Conda and uses the MSVC compiler from
Visual Studio. Other Python dependencies use the same root `requirements.txt`
as Linux.

From PowerShell in the Windows checkout, install all inference components:

```powershell
.\setup.ps1 -NewEnv -Basic -PyTorch3D -FlexGemm -DiffOctreeRast -NvDiffRast -CuMesh -OVoxel -MapAnything -NvDiffRec
```

`-NewEnv` creates and activates the `plsr` Conda environment, then installs
PyTorch and TorchVision. `-Basic` installs the Python requirements, build tools,
Triton Windows, and CUDA Toolkit if it is missing from the environment.
If you already prepared a Conda environment with PyTorch, activate it and omit
`-NewEnv`. Run the script in your current PowerShell session so environment
activation and build settings remain available afterward. The installer saves
SDPA settings in the selected Conda environment and checks activation in a new
Conda process at the end.
If PowerShell blocks scripts, allow them for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The package options can be combined or run individually to resume an install:

| Option | Component |
| --- | --- |
| `-PyTorch3D` | KNN operations for local attention |
| `-FlexGemm` | Sparse convolution and voxel sampling |
| `-DiffOctreeRast` | Patch visibility masks |
| `-NvDiffRast` | Texture baking and material sampling |
| `-CuMesh` | Mesh processing and export |
| `-OVoxel` | Bundled O-Voxel mesh/latent conversion |
| `-MapAnything` | Camera estimation, with torchaudio, utils3d and imageio |
| `-NvDiffRec` | Rendering anchors for camera estimation |

For complete known camera fields, omit `-MapAnything` and `-NvDiffRec`.
The [dependency guide](INFERENCE_DEPENDENCIES.md) explains the input modes.
Windows uses SDPA, so it needs no FlashAttention install option.

Third-party native packages are built from the same pinned revisions as the
Linux installer. DiffOctreeRast uses the public
`JeffreyXiang/diffoctreerast` repository. The camera group also selects
UniCeption 0.1.6 for the optional camera batching adapter.
The script applies the included PyTorch3D, FlexGEMM and cubvh
Windows patches before building their packages. Downloads use pip's normal
cache; cloned native sources are retained under `-BuildRoot` for retries.
Choose a short writable Windows path if PyTorch3D's compiler encounters long
object-file paths, for example:

```powershell
.\setup.ps1 -PyTorch3D -BuildRoot C:\pbuild -MaxJobs 2
```

`-MaxJobs` defaults to 4; lower it if compilation exhausts system RAM.
If Visual Studio auto-detection selects an unsuitable installation, pass
`-Vcvarsall 'C:\path\to\vcvarsall.bat'`.
View all options with `.\setup.ps1 -Help`.

The installer does not load models or run inference. You can separately check
installed package metadata:

```powershell
python scripts\check_inference_environment.py --camera
```

Omit `--camera` when you omitted the camera components.

For inference in a new PowerShell session, activate the environment. SDPA
settings saved by the installer apply automatically:

```powershell
conda activate plsr
```

PLSR defaults to `yxliu/PLSR`; the other models use their official repo IDs.
`PLSR_REPO_ID`, `TRELLIS2_REPO_ID`, `DINO_REPO_ID`, `MAPANYTHING_REPO_ID`, and
`RMBG_REPO_ID` accept a Hugging Face repo ID or a downloaded HF directory.
For example, set `$env:DINO_REPO_ID = 'C:\path\to\dinov3-hf-repo'`.
Set `HF_HOME` to change the Hugging Face download cache location.
See the [inference guide](INFERENCE_USAGE.md) for model loading and input examples.

## Compiler activation

The script checks MSVC and Windows SDK tools before creating an environment or
installing large packages. Installing Visual Studio alone does not provide the
C++ desktop workload or SDK.

Some CUDA 12.8 Conda packages install `vs2017_win-64` with activation scripts
that require VS2017. When VS2022 is selected and this package is present, setup
backs up only `vs2017_compiler_vars.bat` and
`vs2017_get_vsinstall_dir.bat` in the selected environment. The backup names
end in `.bat.plsr-disabled` under `etc/conda/activate.d`; renaming them to the
original `.bat` names restores them. Other environments and hooks are untouched.
Setup stops if both an original hook and its backup already exist, so an earlier
backup is preserved. Reinstalling CUDA compiler packages can restore their hooks;
rerun `-Basic` and check activation again in that case.

The final `conda run ... python --version` check tests a fresh activation rather
than relying on the setup shell's compiler variables. Follow
[VALIDATION.md](VALIDATION.md) for package, GPU and full inference checks.
