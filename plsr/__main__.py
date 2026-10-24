"""PLSR input checks and progressive inference from published model repos."""

import argparse
from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import warnings

from .config import RefinementConfig
from .inputs import LowResolutionImageWarning, validate_inputs
from .pretrained import load_inference_defaults


# Each environment variable accepts a repo ID or a downloaded HF repo directory.
TRELLIS2_REPO_ID = os.environ.get("TRELLIS2_REPO_ID", "microsoft/TRELLIS.2-4B")
PLSR_REPO_ID = os.environ.get("PLSR_REPO_ID", "yxliu/PLSR")
DINO_REPO_ID = os.environ.get("DINO_REPO_ID", "facebook/dinov3-vitl16-pretrain-lvd1689m")
MAPANYTHING_REPO_ID = os.environ.get("MAPANYTHING_REPO_ID", "facebook/map-anything")
RMBG_REPO_ID = os.environ.get("RMBG_REPO_ID", "briaai/RMBG-2.0")


@contextmanager
def _inference_logs_to_stderr():
    """Include native libraries and spawned bake workers in CLI log routing."""
    sys.stdout.flush()
    saved_stdout = os.dup(1)
    try:
        os.dup2(2, 1)
        with redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout, 1)
        os.close(saved_stdout)


def _request_config(args):
    """User paths come from CLI arguments or an optional experiment request."""
    base_dir = Path.cwd()
    data = {}
    if args.config is not None:
        path = args.config.expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Inference request must be a JSON object")
        base_dir = path.parent
    if args.images is not None:
        data["images"] = [{"path": str(path.expanduser().resolve())} for path in args.images]
    for name in ("mesh", "output_dir"):
        value = getattr(args, name)
        if value is not None:
            data[name] = str(value.expanduser().resolve())
    if not data.get("images") or "output_dir" not in data:
        raise ValueError("Supply --image and --output-dir, or provide them in an experiment request JSON")
    for name in ("resolution_sequence", "seed", "decode_intermediate", "output_space"):
        value = getattr(args, name)
        if value is not None:
            data[name] = value
    for role in ("shape", "pbr"):
        guidance = getattr(args, f"{role}_guidance")
        if guidance is not None or args.steps is not None:
            sampler = dict(data.get(f"{role}_sampler", {}))
            if guidance is not None:
                sampler["guidance_strength"] = guidance
            if args.steps is not None:
                sampler["steps"] = args.steps
            data[f"{role}_sampler"] = sampler
    return RefinementConfig.from_dict(data, base_dir, defaults=load_inference_defaults(PLSR_REPO_ID))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m plsr", description="PLSR inference tools")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate inputs without loading models or creating output files")
    infer = commands.add_parser("infer", help="Run PLSR using public model repos or environment overrides")
    for command in (validate, infer):
        command.add_argument("config", nargs="?", type=Path, help="Optional experiment request JSON; paths are relative to this file")
        command.add_argument("--image", dest="images", type=Path, action="append", help="Reference image; repeat for multiple views with a supplied mesh")
        command.add_argument("--mesh", type=Path, help="Coarse mesh; omit to generate it from one image")
        command.add_argument("--output-dir", type=Path, help="Explicit output directory")
        command.add_argument("--resolutions", dest="resolution_sequence", type=int, nargs="+", help="Override the repo's progressive resolution sequence")
        command.add_argument("--shape-guidance", type=float, help="Override shape guidance")
        command.add_argument("--pbr-guidance", type=float, help="Override PBR guidance")
        command.add_argument("--steps", type=int, help="Override sampling steps for both models")
        command.add_argument("--seed", type=int, help="Override the repo's default seed")
        command.add_argument("--decode-intermediate", action="store_true", default=None, help="Also export intermediate meshes")
        command.add_argument("--output-space", choices=("input", "canonical"), help="Override the output coordinate space")
    infer.add_argument("--texture-size", type=int, default=4096, help="Texture size per mesh part (default: 4096)")
    infer.add_argument("--decimation-target", type=int, default=500000, help="Per-part face cap (default: 500000); target is min(cap, face_count/5)")
    infer.add_argument("--low-memory", action="store_true", help="Use smaller batches and camera CPU offload; retain resolutions, steps and patch sizes")
    infer.add_argument("--shape-batch-size", type=int, default=None, help="Shape patches per model batch (default: 12)")
    infer.add_argument("--pbr-batch-size", type=int, default=None, help="PBR patches per model batch (default: 8)")
    infer.add_argument("--camera-batch-size", type=int, default=None, help="Image encoding, independent camera-view attention and dense-head batch size (default: adaptive head, all views elsewhere)")
    infer.add_argument("--camera-token-batch-size", type=int, default=None, help="Camera pointwise MLP tokens per batch; retains all global attention (default: unbounded)")
    infer.add_argument("--camera-offload-encoder", action=argparse.BooleanOptionalAction, default=None, help="Offload the camera image backbone to CPU after encoding all views")
    infer.add_argument("--camera-offload-features", action=argparse.BooleanOptionalAction, default=None, help="Stage camera transformer features and completed prediction batches on CPU")
    infer.add_argument("--sampling-patch-size", type=int, default=32, help="SLat patch edge for both DiTs; at most edge^3 cells per patch (default: 32)")
    infer.add_argument("--sampling-patch-overlap", type=int, default=8, help="Overlapping SLat cells between sampling patches (default: 8)")
    infer.add_argument("--decode-local-resolution", type=int, default=1024, help="Local VAE decode grid resolution (default: 1024)")
    infer.add_argument("--decode-patch-size", type=int, default=64, help="Local SLat decode patch edge (default: 64)")
    infer.add_argument("--decode-overlap", type=int, default=2, help="Overlapping SLat cells between decode patches (default: 2)")
    infer.add_argument("--decode-cell-limit", type=int, default=25000, help="Split decode patches above this latent-cell count (default: 25000)")
    infer.add_argument("--cpu-threads", type=int, default=4, help="Main-process Torch CPU threads (default: 4)")
    args = parser.parse_args(argv)

    try:
        if args.command == "infer":
            from .inference import RunOptions, run_inference
            option_factory = RunOptions.for_low_memory if args.low_memory else RunOptions
            options = option_factory(**{name: getattr(args, name) for name in (
                "texture_size", "decimation_target", "shape_batch_size", "pbr_batch_size",
                "camera_batch_size", "camera_token_batch_size",
                "camera_offload_encoder",
                "camera_offload_features",
                "sampling_patch_size", "sampling_patch_overlap", "decode_local_resolution",
                "decode_patch_size", "decode_overlap", "decode_cell_limit", "cpu_threads",
            ) if getattr(args, name) is not None})
            config = _request_config(args)
            with _inference_logs_to_stderr():
                result = run_inference(config, PLSR_REPO_ID, trellis2_repo_id=TRELLIS2_REPO_ID,
                                       dino_repo_id=DINO_REPO_ID, mapanything_repo_id=MAPANYTHING_REPO_ID,
                                       rmbg_repo_id=RMBG_REPO_ID,
                                       options=options)
            print(json.dumps({"status": "complete", "mesh": str(result.mesh_path),
                              "latents": [str(path) for path in result.latent_paths], "report": str(result.report_path)}, indent=2))
            return 0
        config = _request_config(args)
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter("always", LowResolutionImageWarning)
            report = validate_inputs(config)
        result = {
            "status": "inputs_valid",
            "checks": ["configuration", "mesh_file_readable", "reference_images_decodable"],
            "input": asdict(report),
            "config": config.to_dict(),
            "warnings": [str(message.message) for message in messages],
        }
        if config.mesh is None:
            result["checks"].remove("mesh_file_readable")
        print(json.dumps(result, indent=2, default=str, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
