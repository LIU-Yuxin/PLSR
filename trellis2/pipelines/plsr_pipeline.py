"""Progressive PLSR inference from input assets to exported GLB meshes."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from time import perf_counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from plsr.config import ReferenceImage, RefinementConfig
from plsr.geometry import canonicalize_mesh
from plsr.inputs import validate_inputs
from .reference_images import prepare_reference_images
from ..utils.mesh_export import assemble_parts
from ..utils.slat_io import load_latents, save_latents


@dataclass(frozen=True)
class MeshInferenceResult:
    """Paths to the final mesh, per-level latent archives and run report."""
    mesh_path: Path
    latent_paths: tuple[Path, ...]
    report_path: Path


def _save_glb(scene, path):
    """Export through a temporary GLB and replace the target after success."""
    temporary = path.with_name(f".{path.name}")
    try:
        scene.export(temporary, file_type="glb", extension_webp=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class PLSRPipeline:
    """Run progressive Shape and PBR super-resolution and export GLB meshes.

    Prepare the mesh and reference views, encode the base latents, then reuse
    a PLSRRefinementStep for each requested resolution doubling. The pipeline
    owns the resolution loop, latent files, optional stages and mesh export.

    codec and refinement are already constructed components; their loaded
    models are reused throughout the run. from_pretrained assembles them from
    the model repositories. The refinement step owns single-level computation,
    while the codec owns mesh/latent conversion.
    """

    def __init__(self, codec, refinement, *, camera_estimator=None, coarse_generator=None, image_aligner=None,
                 inference_defaults=None):
        self.codec = codec
        self.refinement = refinement
        self.camera_estimator = camera_estimator
        self.coarse_generator = coarse_generator
        self.image_aligner = image_aligner
        self.inference_defaults = deepcopy({} if inference_defaults is None else inference_defaults)

    @classmethod
    def from_pretrained(cls, plsr_repo_id=None, *, trellis2_repo_id=None, dino_repo_id=None,
                        mapanything_repo_id=None, rmbg_repo_id=None):
        """Load models and default inference parameters from the model repos."""
        from .loading import load_inference_pipeline
        from plsr.pretrained import resolve_model_sources

        sources = resolve_model_sources(
            plsr_repo_id, trellis2_repo_id, dino_repo_id=dino_repo_id,
            mapanything_repo_id=mapanything_repo_id, rmbg_repo_id=rmbg_repo_id,
        )
        return load_inference_pipeline(sources, pipeline_class=cls)

    def make_config(self, *, images, output_dir, mesh=None, **overrides):
        """Build a request from user inputs, inheriting the repo's defaults."""
        views = []
        for view in images:
            if isinstance(view, ReferenceImage):
                view = {**asdict(view), "path": str(view.path)}
            elif isinstance(view, (str, Path)):
                view = {"path": str(view)}
            elif isinstance(view, dict):
                view = {**view, "path": str(view["path"])}
            views.append(view)
        return RefinementConfig.from_dict(
            {**overrides, "images": views, "output_dir": str(output_dir),
             "mesh": None if mesh is None else str(mesh)},
            defaults=self.inference_defaults,
        )

    def prepare_stages(self, config, inputs):
        """Load auxiliary models only when this request needs them."""
        if hasattr(self, "model_sources"):
            from .loading import load_optional_stages

            load_optional_stages(self, config, inputs)

    @torch.no_grad()
    def run(self, config=None, *, images=None, mesh=None, output_dir=None,
            remove_background=None, texture_size=4096,
            decimation_target=500000, shape_batch_size=12, pbr_batch_size=8,
            camera_batch_size=None, camera_token_batch_size=None, camera_offload_encoder=False,
            camera_offload_features=False,
            sampling_patch_size=32, sampling_patch_overlap=8,
            decode_local_resolution=1024, decode_patch_size=64,
            decode_overlap=2, decode_cell_limit=25000,
            verbose=True, runtime_info=None, model_load_seconds=None,
            **inference_overrides):
        """Encode once, refine through the requested levels, and export meshes.

        Each refinement step receives the preceding Shape/PBR latent pair.
        Decode the final level and any requested previews; save latents and run
        records under output_dir. Existing latent archives and GLBs are reused
        automatically; start in a new directory to recompute them. Return their
        paths in MeshInferenceResult.
        """

        # 1. Resolve the request; existing outputs determine which stages are needed.
        if config is None:
            if images is None or output_dir is None:
                raise ValueError("Supply images and output_dir explicitly")
            config = self.make_config(images=images, output_dir=output_dir, mesh=mesh, **inference_overrides)
        elif images is not None or mesh is not None or output_dir is not None or inference_overrides:
            raise ValueError("Supply either a RefinementConfig or inputs and parameter overrides")
        if not isinstance(config, RefinementConfig):
            raise TypeError("config must be a RefinementConfig")

        # 2. Validate execution controls and prepare the output directory.
        output = config.output_dir
        if texture_size is None:
            texture_size = 4096
        if type(texture_size) is not int or texture_size <= 0:
            raise ValueError("texture_size must be a positive integer or None")
        for size in (shape_batch_size, pbr_batch_size):
            if type(size) is not int or size <= 0:
                raise ValueError("Texture and sampler batch sizes must be positive integers")
        for size in (camera_batch_size, camera_token_batch_size):
            if size is not None and (type(size) is not int or size <= 0):
                raise ValueError("Camera batch sizes must be positive integers or None")
        if type(camera_offload_encoder) is not bool:
            raise ValueError("Camera encoder offload must be a boolean")
        if type(camera_offload_features) is not bool:
            raise ValueError("Camera feature offload must be a boolean")
        if type(sampling_patch_size) is not int or sampling_patch_size < 2:
            raise ValueError("sampling_patch_size must be an integer >= 2")
        if type(sampling_patch_overlap) is not int or not 0 <= sampling_patch_overlap < sampling_patch_size:
            raise ValueError("sampling_patch_overlap must be in [0, sampling_patch_size)")
        if (type(decode_local_resolution) is not int or decode_local_resolution <= 0
                or decode_local_resolution % 128):
            raise ValueError("decode_local_resolution must be a positive multiple of 128")
        if (type(decode_patch_size) is not int or decode_patch_size < 2 or decode_patch_size % 2
                or decode_patch_size > decode_local_resolution // 16):
            raise ValueError("decode_patch_size must be even and fit the local SLat grid")
        if type(decode_overlap) is not int or not 0 <= decode_overlap < decode_patch_size:
            raise ValueError("decode_overlap must be in [0, decode_patch_size)")
        if type(decode_cell_limit) is not int or decode_cell_limit <= 0:
            raise ValueError("decode_cell_limit must be a positive integer")
        if decimation_target is not None and (type(decimation_target) is not int or decimation_target <= 0):
            raise ValueError("decimation_target must be a positive integer or None")
        output.mkdir(parents=True, exist_ok=True)

        # Run bookkeeping: settings, completed levels and resumable artifacts.
        def write_json(name, data):
            """Replace one run record after writing a complete JSON document."""
            target = output / name
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(target)

        # Records describe this invocation; existing artifacts are reused by filename.
        write_json("config.json", config.to_dict())
        report = {"status": "running", "levels": [], "output_space": config.output_space,
                  "input_mode": config.input_mode,
                  "input_space": "supplied_mesh" if config.mesh is not None else "generated_coarse_glb",
                  "decode": {"local_resolution": decode_local_resolution, "patch_size": decode_patch_size,
                             "overlap": decode_overlap, "cell_limit": decode_cell_limit},
                  "texture_size": texture_size, "decimation_target": decimation_target,
                  "postprocess": {"remesh": True, "remesh_band": 1, "remesh_project": 0},
                  "shape_batch_size": shape_batch_size, "pbr_batch_size": pbr_batch_size,
                  "camera_batch_size": camera_batch_size,
                  "camera_token_batch_size": camera_token_batch_size,
                  "camera_offload_encoder": camera_offload_encoder,
                  "camera_offload_features": camera_offload_features,
                  "sampling_patch_size": sampling_patch_size,
                  "sampling_patch_overlap": sampling_patch_overlap,
                  "decode_events": "decode_events.jsonl", "timings_log": "timings.jsonl"}
        sequence = config.resolution_sequence
        cached = {resolution for resolution in sequence
                  if (output / f"latents_{resolution}.npz").is_file()}
        required_meshes = set(sequence[1:] if config.decode_intermediate else sequence[-1:])
        completed_meshes = {resolution for resolution in required_meshes
                            if (output / f"mesh_{resolution}.glb").is_file()}
        # Start at the latest available latent. Earlier previews, when requested,
        # require walking the full sequence instead.
        start_index = 0 if config.decode_intermediate else max(
            (index for index, resolution in enumerate(sequence) if resolution in cached), default=0)
        needs_encoding = start_index == 0 and sequence[0] not in cached
        needs_refinement = any(resolution not in cached for resolution in sequence[start_index + 1:])
        paths = []
        transform = None

        # Output helpers keep diagnostics and per-level persistence separate from SR.
        def timing_event(stage, duration_seconds, *, resolution=None, status="complete", **details):
            """Append timing and resource information for one pipeline stage."""
            record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                      "stage": stage,
                      "resolution": resolution, "duration_seconds": round(duration_seconds, 6),
                      "status": status, **details}
            with (output / report["timings_log"]).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

        @contextmanager
        def timed(stage, *, resolution=None):
            """Record elapsed time and peak CUDA memory for a named stage."""
            cuda_active = torch.cuda.is_initialized()
            memory = {}
            if cuda_active:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                memory["cuda_start_allocated_bytes"] = torch.cuda.memory_allocated()
            start = perf_counter()
            status, error = "complete", {}
            try:
                yield
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            except Exception as exc:
                status, error = "failed", {"error": f"{type(exc).__name__}: {exc}"}
                raise
            finally:
                if torch.cuda.is_initialized():
                    memory["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    memory["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                timing_event(stage, perf_counter() - start, resolution=resolution,
                             status=status, **memory, **error)

        if model_load_seconds is not None:
            timing_event("model_loading", model_load_seconds)

        def decode_event(stage, *, resolution, **details):
            """Append a decode or bake event tagged with its resolution."""
            record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                      "resolution": resolution,
                      "stage": stage, **details}
            with (output / report["decode_events"]).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

        def store_level(shape, pbr, resolution, *, reused=False):
            """Save a newly computed pair or register an existing latent archive."""
            path = output / f"latents_{resolution}.npz"
            if not reused:
                with timed("latent_save", resolution=resolution):
                    save_latents(path, shape, pbr, resolution)
            paths.append(path)
            level = {"resolution": resolution, "slat_cells": len(shape.coords),
                     "latents": path.name, "reused": reused}
            report["levels"].append(level)
            write_json("report.json", report)
            return level

        def export_level(shape, pbr, resolution, level):
            """Decode and bake one level, then register its GLB in the run report."""
            path = output / f"mesh_{resolution}.glb"
            if path.is_file():
                level["mesh"] = path.name
                timing_event("mesh_cache_reuse", 0.0, resolution=resolution)
                write_json("report.json", report)
                return path
            if verbose:
                print(f"Decoding and exporting resolution {resolution}", flush=True)
            part_count = None

            def log(stage, **details):
                nonlocal part_count
                decode_event(stage, resolution=resolution, **details)
                if stage == "decode_plan":
                    part_count = details["parts"]
                    if verbose:
                        print(f"Decode/export {resolution}: {part_count} parts", flush=True)
                elif verbose:
                    part = details.get("part", 0) + 1
                    progress = f"Decode/export {resolution}, part {part}/{part_count}"
                    if stage == "part_extracted":
                        print(f"{progress}: decoding Shape/PBR", flush=True)
                    elif stage == "part_bake_start":
                        print(f"{progress}: remeshing and baking textures", flush=True)
                    elif stage == "part_merged":
                        print(f"{progress}: complete", flush=True)
                    elif stage == "glb_export_start":
                        print(f"Writing mesh_{resolution}.glb", flush=True)
                    elif stage == "export_complete":
                        print(f"Exported mesh_{resolution}.glb ({details['bytes'] / 2**20:.1f} MiB)", flush=True)

            log("export_start", latent_cells=len(shape.coords))
            try:
                scene = assemble_parts(
                    self.codec.decode_parts(shape, pbr, resolution,
                                            local_resolution=decode_local_resolution,
                                            patch_size=decode_patch_size, overlap=decode_overlap,
                                            cell_limit=decode_cell_limit, on_event=log),
                    canonical_to_input=transform, texture_size=texture_size,
                    decimation_target=decimation_target, verbose=verbose, on_event=log,
                )
                path = output / f"mesh_{resolution}.glb"
                log("glb_export_start", scene_parts=len(scene.geometry), file=path.name)
                _save_glb(scene, path)
                log("export_complete", scene_parts=len(scene.geometry),
                    file=path.name, bytes=path.stat().st_size)
            except Exception as exc:
                log("export_failed", error=f"{type(exc).__name__}: {exc}")
                raise
            level.update(mesh=path.name, parts=len(scene.geometry))
            write_json("report.json", report)
            return path

        write_json("report.json", report)
        run_start = perf_counter()
        try:
            # Completed meshes are sufficient; no cache metadata or input reads
            # are required when every requested export is already present.
            if required_meshes <= completed_meshes:
                for resolution in sequence:
                    level = {"resolution": resolution, "reused": True}
                    if resolution in cached:
                        path = output / f"latents_{resolution}.npz"
                        paths.append(path)
                        level["latents"] = path.name
                    mesh = output / f"mesh_{resolution}.glb"
                    if mesh.is_file():
                        level["mesh"] = mesh.name
                    if "latents" in level or "mesh" in level:
                        report["levels"].append(level)
                report["status"] = "complete"
                write_json("report.json", report)
                return MeshInferenceResult(output / f"mesh_{sequence[-1]}.glb", tuple(paths), output / "report.json")

            asset = None
            transform_path = output / "transforms.json"
            needs_source = needs_encoding or needs_refinement or (
                config.output_space == "input" and not transform_path.is_file())
            if needs_source:
                inputs = validate_inputs(config)
                self.prepare_stages(config, inputs)
                if self.camera_estimator is not None:
                    configure_camera = getattr(self.camera_estimator, "configure_batches", None)
                    if configure_camera is not None:
                        configure_camera(camera_batch_size, camera_token_batch_size, camera_offload_encoder, camera_offload_features)
                    elif camera_batch_size is not None or camera_token_batch_size is not None or camera_offload_encoder or camera_offload_features:
                        raise ValueError("This camera estimator does not support batching controls")
                if remove_background is None:
                    remove_background = getattr(self, "remove_background", None)
                if config.mesh is None and not (output / "coarse.glb").is_file() and self.coarse_generator is None:
                    raise ValueError("Single-image inference requires an original TRELLIS.2 coarse generator")
                if needs_refinement and config.needs_camera_estimation and self.camera_estimator is None:
                    raise ValueError("Missing camera fields require a camera estimator")
                # 3. Use the supplied mesh, or generate and export a TRELLIS.2 coarse mesh.
                mesh_source = config.mesh
                if mesh_source is None:
                    mesh_source = output / "coarse.glb"
                    if mesh_source.is_file():
                        timing_event("coarse_cache_reuse", 0.0, resolution=1024)
                        report.setdefault("coarse", {})["mesh"] = mesh_source.name
                    else:
                        latent_path = output / "coarse_latents.npz"
                        condition_path = output / "coarse_condition.png"
                        if latent_path.is_file() and condition_path.is_file():
                            from .trellis2_coarse import CoarseLatents
                            with timed("coarse_cache_load", resolution=1024):
                                coarse_shape, coarse_pbr, coarse_resolution = load_latents(latent_path)
                                with Image.open(condition_path) as condition:
                                    coarse = CoarseLatents(coarse_shape, coarse_pbr, condition.copy(),
                                                           coarse_resolution)
                        else:
                            write_json("coarse_settings.json", self.coarse_generator.settings)
                            if verbose:
                                print("Generating the original TRELLIS.2 coarse mesh at resolution 1024", flush=True)
                            with timed("coarse_generation", resolution=1024):
                                with Image.open(config.images[0].path) as image:
                                    coarse = self.coarse_generator.run(image.copy(), seed=config.seed, verbose=verbose)
                            with timed("coarse_latent_save", resolution=coarse.resolution):
                                save_latents(latent_path, coarse.shape, coarse.pbr, coarse.resolution)
                            coarse.condition_image.save(condition_path)
                        report["coarse"] = {"resolution": coarse.resolution, "slat_cells": len(coarse.shape.coords),
                                            "latents": latent_path.name, "condition_image": condition_path.name,
                                            "settings": "coarse_settings.json"}
                        write_json("report.json", report)
                        from .trellis2_coarse import Trellis2CoarsePipeline
                        coarse_log = lambda stage, **details: decode_event(stage, resolution=coarse.resolution,
                                                                            phase="coarse", **details)
                        coarse_log("export_start", latent_cells=len(coarse.shape.coords))
                        try:
                            with timed("coarse_decode_export", resolution=coarse.resolution):
                                if isinstance(self.coarse_generator, Trellis2CoarsePipeline):
                                    coarse_scene = self.coarse_generator.export(coarse, verbose=verbose,
                                                                                on_event=coarse_log)
                                else:
                                    coarse_scene = self.coarse_generator.export(coarse, verbose=verbose)
                                _save_glb(coarse_scene, mesh_source)
                        except Exception as exc:
                            coarse_log("export_failed", error=f"{type(exc).__name__}: {exc}")
                            raise
                        coarse_log("export_complete", scene_parts=len(coarse_scene.geometry),
                                   file=mesh_source.name, bytes=mesh_source.stat().st_size)
                        report["coarse"]["mesh"] = mesh_source.name
                        del coarse
                        write_json("report.json", report)

                # 4. Rotate and normalize the source mesh, retaining the inverse transform.
                # Coarse latents are not reused directly: SR starts by encoding this mesh.
                with timed("mesh_canonicalization"):
                    asset = canonicalize_mesh(mesh_source)
                report["base_voxelization"] = {
                    "mode": "independent_coarse_parts" if config.mesh is None else "whole_scene",
                    "source_parts": len(asset.scene.geometry),
                }
                write_json("report.json", report)
                transform = asset.canonical_to_input if config.output_space == "input" else None
                write_json("transforms.json", {
                    "input_to_canonical": asset.input_to_canonical.tolist(),
                    "canonical_to_input": asset.canonical_to_input.tolist(),
                })

            elif config.output_space == "input":
                # Final-latent export needs only the saved coordinate transform.
                transform = np.asarray(json.loads(transform_path.read_text())["canonical_to_input"])

            if needs_refinement:
                # 5. Prepare reference images and fill missing camera fields.
                # Estimated-camera images are matched to the coarse silhouette;
                # the target camera matrices stay fixed during image alignment.
                with timed("reference_preparation"):
                    aligner = self.image_aligner
                    if config.needs_camera_estimation and aligner is None:
                        from .reference_alignment import RenderedReferenceAligner
                        aligner = RenderedReferenceAligner(self.codec.device)
                    views = prepare_reference_images(config.images, asset.input_to_canonical, remove_background=remove_background,
                                                      estimate_cameras=self.camera_estimator, canonical_scene=asset.scene,
                                                      align_images=aligner)
                    alignment_records = []
                    for index, alignment in views.alignments:
                        image_name, mask_name = f"reference_{index}_aligned.png", f"reference_{index}_target_alpha.png"
                        alignment.image.save(output / image_name)
                        alignment.target_alpha.save(output / mask_name)
                        alignment_records.append({"view": index, "matrix": alignment.matrix.tolist(),
                                                  "image": image_name, "target_alpha": mask_name})
                    write_json("image_alignment.json", {
                        "method": "silhouette_centroid_and_spread", "resampling": "nearest",
                        "matrix_maps": "prepared full reference pixels to target camera canvas pixels",
                        "camera_matrices_changed": False, "views": alignment_records,
                        "rgba_convention": "RGB already composited onto black; multiply by alpha again for SR",
                    })
                    write_json("cameras.json", {
                        "space": "canonical", "convention": "OpenCV world-to-camera", "intrinsics_units": "normalized",
                        "image_size": list(views.images.shape[-2:]),
                        "intrinsics": views.intrinsics.tolist(), "extrinsics": views.extrinsics.tolist(),
                        "estimated_fields": [list(view.missing_camera_fields) for view in config.images],
                    })


            # 6. Restore the latest available latent, or encode the base once.
            # Earlier saved levels remain listed without decompressing them.
            for resolution in sequence[:start_index]:
                if resolution in cached:
                    path = output / f"latents_{resolution}.npz"
                    paths.append(path)
                    level = {"resolution": resolution, "latents": path.name, "reused": True}
                    mesh = output / f"mesh_{resolution}.glb"
                    if mesh.is_file():
                        level["mesh"] = mesh.name
                    report["levels"].append(level)
            resolution = sequence[start_index]
            if resolution in cached:
                with timed("latent_cache_load", resolution=resolution):
                    shape, pbr, _ = load_latents(output / f"latents_{resolution}.npz")
                stage = "base_encoding_reused" if start_index == 0 else "refinement_reused"
                timing_event(stage, 0.0, resolution=resolution)
                level = store_level(shape, pbr, resolution, reused=True)
            else:
                if verbose:
                    print(f"Encoding base mesh at resolution {resolution}", flush=True)
                with timed("base_encoding", resolution=resolution):
                    shape, pbr = self.codec.encode(asset.scene, resolution, independent_parts=config.mesh is None)
                level = store_level(shape, pbr, resolution)
            del asset
            mesh_path = None
            if resolution in required_meshes:
                with timed("decode_export", resolution=resolution):
                    mesh_path = export_level(shape, pbr, resolution, level)

            # 7. Continue from each level's latents; cached levels are read once.
            for resolution in sequence[start_index + 1:]:
                if resolution in cached:
                    with timed("latent_cache_load", resolution=resolution):
                        shape, pbr, _ = load_latents(output / f"latents_{resolution}.npz")
                    timing_event("refinement_reused", 0.0, resolution=resolution)
                    level = store_level(shape, pbr, resolution, reused=True)
                else:
                    if verbose:
                        print(f"Refining Shape and PBR to resolution {resolution}", flush=True)
                    with timed("refinement", resolution=resolution):
                        refined = self.refinement.run(
                            shape, pbr, views.images, views.intrinsics, views.extrinsics,
                            resolution=resolution, shape_sampler_params=config.shape_sampler,
                            pbr_sampler_params=config.pbr_sampler, seed=config.seed,
                            shape_batch_size=shape_batch_size, pbr_batch_size=pbr_batch_size, verbose=verbose,
                            sampling_patch_size=sampling_patch_size,
                            sampling_patch_overlap=sampling_patch_overlap,
                            on_timing=lambda stage, seconds: timing_event(stage, seconds, resolution=resolution),
                        )
                    if refined.resolution != resolution:
                        raise ValueError("Refinement returned an unexpected resolution")
                    shape, pbr = refined.shape, refined.pbr
                    del refined
                    level = store_level(shape, pbr, resolution)
                # Mesh previews are outputs only; subsequent SR keeps the latents.
                if resolution in required_meshes:
                    with timed("decode_export", resolution=resolution):
                        mesh_path = export_level(shape, pbr, resolution, level)

            # 8. Finalize the run after the final mesh has been exported.
            if mesh_path is None:
                raise RuntimeError("Inference finished without an exported mesh")
            report["status"] = "complete"
            write_json("report.json", report)
        except Exception as exc:
            # Completed files remain available to the next ordinary invocation.
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            write_json("report.json", report)
            raise
        finally:
            if runtime_info is not None:
                write_json("runtime.json", runtime_info)
            timing_event("pipeline_total", perf_counter() - run_start, status=report["status"])
        return MeshInferenceResult(mesh_path, tuple(paths), output / "report.json")
