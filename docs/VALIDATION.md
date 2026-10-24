# Fresh-machine validation

Validate the standalone inference directory on a new environment. Use the same
source snapshot on Linux and Windows; retain the user's CUDA architecture
settings. Do not copy compiled extensions, Conda activation edits, or caches
from the development machine into the release.

## Install and open a new shell

Follow [INSTALL.md](INSTALL.md) or [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md).
Record OS, Python, GPU/driver, CUDA Toolkit and selected attention backends.
The installers retain architecture settings such as `TORCH_CUDA_ARCH_LIST` and
`FLASH_ATTN_CUDA_ARCHS`. A slower multi-architecture build is not a reason to
silently narrow those targets.

Close the installation shell and activate the new environment in a new terminal.
On Windows, confirm that activation works without the setup shell's MSVC
variables, and that the saved attention backends are SDPA.

Run from the standalone release root:

```text
python -m pip check
python scripts/check_inference_environment.py --camera
python scripts/check_cuda_extensions.py --camera
python scripts/check_camera_batches.py --device cpu
python scripts/check_camera_batches.py --device cuda
python -m plsr infer --help
```

Omit camera checks only for an intentionally camera-free installation using a
supplied mesh with complete intrinsics and extrinsics. The camera comparison
uses small upstream attention/MLP blocks and an encoder, without weights or
network access. It checks both MLP variants, positional inputs, register order,
weight preservation and restoration of the default graph. It does not replace
full MapAnything or full pipeline validation.

## Run the complete pipeline

Use a fresh output directory and an image available on the test machine. Select
normal mode or `--low-memory` explicitly according to the available resources;
this is not inferred from the GPU model. For example:

```text
python -m plsr infer --image /path/to/reference.png --output-dir /path/to/result --resolutions 640 1280 2560 --seed 42 --low-memory
```

Replace the paths with local paths; PowerShell also accepts forward slashes.
The normal path uses the same command without `--low-memory`. Compare the two
in separate output directories when resources permit. Check that explicit batch
and offload options override the preset in `runtime.json` and `report.json`.
Preset selection must retain resolutions, views, steps, patch sizes, precision
policy and export quality settings.

For a shorter first test, use `--resolutions 640 1280`; subsequently extend the
same request to 2560. Reuse the same image, seed, models and sampling settings.
The completed latent levels should be reused. Do not reuse an output directory
for a different input or sampling configuration.

Observe all stages: coarse generation, camera anchor rendering, camera
prediction, base encoding, Shape/PBR SR, per-part decoding/baking and final GLB
writing. On Windows also observe dedicated/shared GPU memory in Task Manager;
heavy shared-memory use can cause slowdown without an OOM. Include baking
workers when examining GPU memory. CPU offload also requires system RAM.

Confirm that:

- `report.json` ends in `complete`, and requested latent files and final GLB exist.
- Decode progress shows the total part count, each part's decoding and baking,
  and final GLB writing. Sampling completion alone is not completion of the run.
- The GLB opens with geometry and textures. Inspect detail, overlap boundaries,
  transparency and material appearance; metadata checks do not assess quality.
- Rerunning the same command reuses completed files. Extending the resolution
  sequence retains earlier latents. Existing recovery resumes at latent levels,
  not inside an interrupted sampler or partially baked export.
- A supplied-mesh/known-camera case bypasses camera estimation. When comparing
  renderer changes, include both opaque and transparent materials.

Keep the effective config, runtime/report JSON, timings/decode events and
console log with the result. Report installation checks, numerical checks and
completed inference separately. Include the chosen batches, final latent cell
count, part count, file size and stage timings. A result on one asset does not
establish a universal memory requirement.
