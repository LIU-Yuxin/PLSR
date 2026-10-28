"""Local web interface for image-guided PLSR refinement."""

import argparse
import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4
import zipfile


ROOT = Path(__file__).resolve().parent
EXAMPLE = ROOT / "assets" / "venice_mask"
EXAMPLE_IMAGES = ROOT / "assets" / "images"


def output_directory(path):
    path = Path(path).expanduser().resolve()
    if path == ROOT:
        raise ValueError("Choose an output subdirectory, such as outputs/gradio.")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _number(value, name, minimum, maximum=None, *, integer=False):
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a number.")
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range.")
    if integer and not value.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(value) if integer else value


def prepare_run(output_root, images, mesh, resolution, low_memory, independent,
                min_iou, seed, steps, shape_cfg, pbr_cfg, texture_size, face_cap,
                intermediate, output_space, texture_budget=0):
    """Copy inputs and build a CLI invocation in a fresh output directory."""
    from PIL import Image

    if not images:
        raise ValueError("Upload at least one reference image.")
    if len(images) > 1 and not mesh:
        raise ValueError("Multiple reference images require a coarse GLB model.")
    image_paths = [Path(item[0] if isinstance(item, (tuple, list)) else item) for item in images]
    for path in image_paths:
        with Image.open(path) as image:
            image.verify()
    mesh_path = Path(mesh) if mesh else None
    if mesh_path is not None and (not mesh_path.is_file() or mesh_path.suffix.lower() != ".glb"):
        raise ValueError("Upload a self-contained GLB model.")
    resolution = _number(resolution, "Resolution", 1280, 2560, integer=True)
    if resolution not in (1280, 2560):
        raise ValueError("Final resolution must be 1280 or 2560.")
    if output_space not in ("input", "canonical"):
        raise ValueError("Output coordinates must be input or canonical.")
    config = {
        "resolution_sequence": [640, 1280] if resolution == 1280 else [640, 1280, 2560],
        "seed": _number(seed, "Seed", 0, 2**32 - 1, integer=True),
        "min_view_iou": _number(min_iou, "Minimum view IoU", 0, 1),
        "independent_camera_estimation": bool(independent),
        "decode_intermediate": bool(intermediate),
        "output_space": output_space,
    }
    for role, guidance in (("shape", shape_cfg), ("pbr", pbr_cfg)):
        sampler = {}
        if guidance is not None:
            sampler["guidance_strength"] = _number(guidance, f"{role} CFG", 0)
        if steps is not None:
            sampler["steps"] = _number(steps, "Sampling steps", 1, integer=True)
        if sampler:
            config[f"{role}_sampler"] = sampler
    texture_size = _number(texture_size, "Texture size", 1, integer=True)
    texture_budget = _number(texture_budget, "Texture budget", 0, integer=True)
    if texture_budget:
        for name, value in (("Texture size", texture_size), ("Texture budget", texture_budget)):
            if value < 256 or value & (value - 1):
                raise ValueError(f"{name} must be a power of two >= 256 when budgeting textures.")
    face_cap = _number(face_cap, "Face cap", 1, integer=True)
    run_dir = output_directory(output_root) / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8])
    inputs = run_dir / "inputs"
    inputs.mkdir(parents=True)
    config["images"] = []
    for index, path in enumerate(image_paths):
        destination = inputs / f"view_{index:03d}{path.suffix.lower()}"
        shutil.copyfile(path, destination)
        config["images"].append({"path": str(destination)})
    if mesh_path is not None:
        destination = inputs / "coarse.glb"
        shutil.copyfile(mesh_path, destination)
        config["mesh"] = str(destination)
    config["output_dir"] = str(run_dir)
    config_path = run_dir / "input_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    command = [sys.executable, "-u", "-B", "-m", "plsr", "infer", str(config_path),
               "--texture-size", str(texture_size), "--decimation-target", str(face_cap)]
    if texture_budget:
        command.extend(["--texture-budget", str(texture_budget)])
    if low_memory:
        command.append("--low-memory")
    return run_dir, command


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 16000))
        return stream.read().decode("utf-8", errors="replace")


def diagnostics(run_dir):
    try:
        report = json.loads((run_dir / "view_diagnostics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [(str(run_dir / view["overlay"]),
             f"View {view['view']} | IoU {view['iou']:.1%} | " + ("kept" if view["accepted"] else "rejected"))
            for view in report["views"] if (run_dir / view["overlay"]).is_file()]



def completed_meshes(run_dir):
    """Only preview files acknowledged as complete by the inference report."""
    try:
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    meshes = []
    coarse = report.get("coarse", {})
    if coarse.get("mesh"):
        meshes.append((run_dir / coarse["mesh"], "coarse model"))
    for level in report.get("levels", []):
        if level.get("mesh"):
            meshes.append((run_dir / level["mesh"], f"refined model at {level['resolution']}"))
    return [(path, label) for path, label in meshes if path.is_file()]


def collect_outputs(run_dir, resolution):
    """Keep large model files separate from the small diagnostic archive."""
    archive = run_dir / "diagnostics.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for pattern in ("*.json", "*.jsonl", "*.log", "view_diagnostics/*.png"):
            for path in sorted(run_dir.glob(pattern)):
                bundle.write(path, path.relative_to(run_dir))
    models = [path for path, _ in completed_meshes(run_dir)]
    files = models + sorted(run_dir.glob("latents_*.npz")) + [archive]
    final = run_dir / f"mesh_{int(resolution)}.glb"
    return str(final) if final.is_file() else None, [str(path) for path in files]


def stop_process(process):
    """Stop inference together with any active export workers."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def create_app(output_root):
    output_root = output_directory(output_root)
    os.environ["GRADIO_TEMP_DIR"] = str(output_root / ".gradio")
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    example_paths = sorted(path for path in EXAMPLE_IMAGES.iterdir()
                           if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    def select_image_example(evt: gr.SelectData):
        return gr.Gallery(value=[str(example_paths[evt.index])], selected_index=0), None

    async def infer(*values):
        try:
            run_dir, command = prepare_run(output_root, *values)
        except (OSError, ValueError, TypeError) as exc:
            raise gr.Error(str(exc)) from exc
        yield "Starting inference...", None, [], [], ""
        started = time.monotonic()
        log_path = run_dir / "inference.log"
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        process_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
                           if os.name == "nt" else {"start_new_session": True})
        last_diagnostics = None
        last_models = ()
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, **process_options)
            try:
                while process.poll() is None:
                    report_path = run_dir / "view_diagnostics.json"
                    stamp = report_path.stat().st_mtime_ns if report_path.exists() else None
                    gallery = diagnostics(run_dir) if stamp != last_diagnostics else gr.skip()
                    last_diagnostics = stamp
                    meshes = completed_meshes(run_dir)
                    paths = tuple(str(path) for path, _ in meshes)
                    model_update = paths[-1] if paths and paths != last_models else gr.skip()
                    files_update = list(paths) if paths != last_models else gr.skip()
                    last_models = paths
                    preview = f" | Preview: {meshes[-1][1]}" if meshes else ""
                    elapsed = int(time.monotonic() - started)
                    yield (f"Running{preview} | {elapsed // 60}m {elapsed % 60}s | {run_dir}",
                           model_update, files_update, gallery, log_tail(log_path))
                    await asyncio.sleep(1)
            finally:
                stop_process(process)
        model, files = collect_outputs(run_dir, values[2])
        elapsed = int(time.monotonic() - started)
        success = process.returncode == 0 and model is not None
        meshes = completed_meshes(run_dir)
        status = "Complete" if success else "Failed; see the log and silhouette diagnostics"
        if not success and meshes:
            status += f" | Preview: {meshes[-1][1]}"
        preview = model if success else (str(meshes[-1][0]) if meshes else None)
        yield (f"{status} | {elapsed // 60}m {elapsed % 60}s | {run_dir}",
               preview, files, diagnostics(run_dir), log_tail(log_path))

    def load_example():
        config = json.loads((EXAMPLE / "estimated_cameras.json").read_text(encoding="utf-8"))
        return [str(EXAMPLE / image["path"]) for image in config["images"]], str(EXAMPLE / config["mesh"])

    with gr.Blocks(title="PLSR") as demo:
        gr.Markdown("# PLSR\nRefine geometry and PBR materials from reference images.")
        gr.Markdown("Upload one image to generate and refine a model, or add a coarse GLB. "
                    "Multiple images require a coarse GLB. Cameras are estimated automatically.")
        with gr.Row():
            with gr.Column():
                images = gr.Gallery(label="Reference images", file_types=["image"], type="filepath",
                                    sources=["upload"], columns=3, rows=2, height=280, preview=True,
                                    object_fit="contain", interactive=True)
                with gr.Accordion("Example images", open=True):
                    gr.Markdown("Select an image to load it without a coarse model.")
                    image_examples = gr.Gallery([(str(path), path.stem) for path in example_paths],
                                                show_label=False, columns=4, rows=2, height=220,
                                                object_fit="contain", allow_preview=False, interactive=False)
                mesh = gr.Model3D(label="Coarse GLB (optional for one image)", height=300, interactive=True)
                example = gr.Button("Load Venice mask inputs", size="sm")
                gr.Markdown("[Venice Mask by DailyArt](https://sketchfab.com/3d-models/venice-mask-4aace12762ee44cf97d934a6ced12e65) · CC BY-NC 4.0")
                resolution = gr.Dropdown([1280, 2560], value=1280, label="Final resolution", info="Refinement starts at 640.")
                low_memory = gr.Checkbox(value=True, label="Low memory mode")
                independent = gr.Checkbox(value=False, label="Estimate cameras separately", info="Recommended for multiple images on low-VRAM GPUs. Usually slower; joint estimation is the default.")
                min_iou = gr.Slider(0, 1, value=0.6, step=0.01, label="Minimum view IoU", info="Poorly aligned views are discarded. Stops if no views remain.")
                with gr.Accordion("Sampling and export", open=False):
                    seed = gr.Number(value=42, precision=0, label="Seed", minimum=0, maximum=2**32 - 1)
                    steps = gr.Number(value=None, precision=0, minimum=1, label="Sampling steps", info="Leave blank to use model defaults.")
                    shape_cfg = gr.Number(value=None, minimum=0, label="Shape CFG", info="Leave blank to use model defaults.")
                    pbr_cfg = gr.Number(value=None, minimum=0, label="PBR CFG", info="Leave blank to use model defaults.")
                    texture_size = gr.Dropdown([1024, 2048, 4096], value=4096, label="Texture size per chunk")
                    texture_budget = gr.Dropdown([("Off", 0), ("4096", 4096), ("8192", 8192), ("16384", 16384)],
                                                 value=0, label="Object texture budget",
                                                 info="Limits total texture pixels to this value squared per material map. Smaller textures may lose fine detail.")
                    face_cap = gr.Number(value=500000, precision=0, minimum=1, label="Face cap per part")
                    intermediate = gr.Checkbox(value=False, label="Export intermediate resolutions")
                    output_space = gr.Dropdown(["input", "canonical"], value="input", label="Output coordinates")
                with gr.Row():
                    run = gr.Button("Run", variant="primary")
                    stop = gr.Button("Stop")
            with gr.Column():
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
                model = gr.Model3D(label="Model preview (coarse to refined)", height=480)
                downloads = gr.File(label="Download results", file_count="multiple", interactive=False)
                gallery = gr.Gallery(label="Silhouette overlap", columns=2, height=420,
                                     object_fit="contain", interactive=False)
                gr.Markdown("Cyan: image only · Orange: mesh only · Light gray: overlap. "
                            "Overlays use the centered cameras and aligned reference images.")
                with gr.Accordion("Run log", open=False):
                    logs = gr.Textbox(label="Recent output", lines=14, max_lines=24, interactive=False)
        inputs = [images, mesh, resolution, low_memory, independent, min_iou, seed, steps,
                  shape_cfg, pbr_cfg, texture_size, face_cap, intermediate, output_space, texture_budget]
        event = run.click(infer, inputs, [status, model, downloads, gallery, logs],
                          concurrency_limit=1, concurrency_id="inference", api_name="infer")
        stop.click(lambda: "Stop requested. Saved files remain in the output directory.",
                   outputs=status, cancels=[event], queue=False, api_name="stop")
        example.click(load_example, outputs=[images, mesh], queue=False, api_name="example")
        image_examples.select(select_image_example, outputs=[images, mesh], queue=False, api_name=False)
    return demo.queue(default_concurrency_limit=1)


def main():
    parser = argparse.ArgumentParser(description="Start the local PLSR web interface.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "gradio",
                        help="Uploads and results directory (default: outputs/gradio)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    output_root = output_directory(args.output_dir)
    try:
        demo = create_app(output_root)
    except ModuleNotFoundError as exc:
        if exc.name != "gradio":
            raise
        parser.error("Install the optional interface: pip install -r requirements-gradio.txt")
    demo.launch(server_name=args.host, server_port=args.port, share=False,
                allowed_paths=[str(output_root), str(EXAMPLE), str(EXAMPLE_IMAGES)])


if __name__ == "__main__":
    main()
