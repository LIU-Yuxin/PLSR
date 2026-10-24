"""Missing-camera estimation with rendered PBR anchors and MapAnything."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from ..utils.camera_alignment import align_camera_centers, align_extrinsics, remove_rotation_scale


def prepare_mapanything_views(images, intrinsics, extrinsics):
    """Pad to squares and track the upstream pixel-camera resize transformation.

    Intrinsics are normalized to each unpadded input image. A dummy calibration
    is used only to measure the image transform and is removed before inference
    for uncalibrated views. It never becomes a model condition.
    """
    from mapanything.utils.image import preprocess_inputs
    from mapanything.utils.cropping import crop_resize_if_necessary

    if not images or len(images) != len(intrinsics) or len(images) != len(extrinsics):
        raise ValueError("Images and camera lists must have matching nonzero lengths")
    squares, calibrations, transforms = [], [], []
    for image, calibration in zip(images, intrinsics):
        width, height = image.size
        side = max(width, height)
        left, top = (side - width) // 2, (side - height) // 2
        square = Image.new("RGB", (side, side))
        square.paste(image.convert("RGB"), (left, top))
        squares.append(square)
        padding = np.eye(3)
        padding[:2, 2] = [left, top]
        # Upstream accounts for pixel-center conventions as well as dimensions.
        _, resize = crop_resize_if_necessary(square, (518, 518), intrinsics=np.eye(3))
        transform = resize @ padding
        transforms.append(transform)
        matrix = None
        if calibration is not None:
            matrix = np.asarray(calibration).copy()
            matrix[0] *= width
            matrix[1] *= height
            matrix = transform @ matrix
        calibrations.append(matrix)
    # The upstream in-memory API accepts PIL images and applies DINOv2 normalization.
    processed = preprocess_inputs(
        [{"img": image} for image in squares], resize_mode="fixed_mapping",
        resolution_set=518, norm_type="dinov2", verbose=False,
    )
    if len(processed) != len(images):
        raise ValueError("MapAnything preprocessing did not preserve every reference view")
    for index, (view, calibration, extrinsic) in enumerate(zip(processed, calibrations, extrinsics)):
        view.update(true_shape=np.int32([view["img"].shape[-2:]]), idx=index, instance=str(index))
        if calibration is not None:
            view["intrinsics"] = torch.from_numpy(calibration).float()[None]
        if extrinsic is not None:
            view["camera_poses"] = torch.linalg.inv(torch.as_tensor(extrinsic, dtype=torch.float32))[None]
            view["is_metric_scale"] = torch.tensor([True])
    return processed, transforms


class MapAnythingCameraEstimator:
    """Estimate missing reference cameras with mesh-space render anchors.

    Each of two rounds renders calibrated anchors, predicts a joint camera
    reconstruction, then fits a Sim(3) from predicted to known anchor centers.
    Round two adds views around the first estimates. Supplied fields are kept;
    only inferred focal lengths receive the object-framing cap. Input scenes
    and returned world-to-camera poses use the same canonical mesh coordinates.
    """

    def __init__(self, model, *, envmap_path=None, device="cuda", low_vram=True):
        """Bind a loaded model, bundled anchor lighting and stage offload policy."""
        self.model = model.eval()
        self.device = torch.device(device)
        self.low_vram = low_vram
        # Radius is measured in the normalized mesh coordinate system.
        self.anchor_radius = 2.8
        self.envmap_path = Path(envmap_path) if envmap_path is not None else Path(__file__).resolve().parents[2] / "assets/hdri/forest.exr"
        self.last_report = []
        self.minibatch_size = None

    def configure_batches(self, view_batch_size=None, token_batch_size=None, offload_encoder=False, offload_features=False):
        """Keep all camera views; batch independent attention and MLP work."""
        from .camera_batches import configure_mapanything_batches

        configure_mapanything_batches(self.model, view_batch_size, token_batch_size, offload_encoder, offload_features)
        self.minibatch_size = view_batch_size

    def _anchors(self, previous):
        """Build 77 hemisphere views plus 24 local neighbors per previous pose."""
        from ..utils.render_utils import yaw_pitch_r_fov_to_extrinsics_intrinsics, yaw_pitch_r_from_extrinsics

        yaws, pitches = [], []
        for pitch, count in zip((10, 20, 30), (41, 23, 13)):
            for index in range(count):
                yaws.append(index * 2 * np.pi / count - 16 / 180 * np.pi)
                pitches.append(pitch / 180 * np.pi)
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches, self.anchor_radius, 30.)
        anchors = list(zip(extrinsics, intrinsics))
        for extrinsic in previous:
            yaw, pitch, radius = yaw_pitch_r_from_extrinsics(extrinsic)
            fov = np.arctan(.6 / radius) * 2 * 180 / np.pi
            offsets = [(x, y) for x in (-10, -5, 0, 5, 10) for y in (-10, -5, 0, 5, 10) if x or y]
            ext, intr = yaw_pitch_r_fov_to_extrinsics_intrinsics(
                [yaw + x / 180 * np.pi for x, _ in offsets],
                [pitch + y / 180 * np.pi for _, y in offsets], radius, fov,
            )
            anchors = list(zip(ext, intr)) + anchors
        return anchors

    def _render_anchors(self, scene, anchors):
        """Render 518-pixel PBR anchors with SSAA and bundled forest lighting."""
        import cv2
        from torchvision.transforms.functional import to_pil_image
        from ..renderers import EnvMap
        from ..renderers.pbr_mesh_renderer import PbrMeshRenderer
        from ..utils.pbr_scene import scene_to_pbr_mesh

        hdr = cv2.imread(str(self.envmap_path), cv2.IMREAD_UNCHANGED)
        if hdr is None:
            raise ValueError(f"Cannot read anchor environment map: {self.envmap_path}")
        environment = EnvMap(torch.tensor(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB), dtype=torch.float32, device=self.device))
        mesh = scene_to_pbr_mesh(scene, self.device)
        renderer = PbrMeshRenderer({"resolution": 518, "ssaa": 2, "near": .01, "far": 100., "peel_layers": 8})
        images = []
        for index, (ext, intr) in enumerate(anchors):
            rendered = renderer.render(mesh, ext.to(self.device), intr.to(self.device), environment)
            images.append(to_pil_image(rendered["shaded"].cpu()))
            del rendered
            if (index + 1) % 10 == 0 or index + 1 == len(anchors):
                print(f"Rendered camera anchors: {index + 1}/{len(anchors)}", flush=True)
        return images

    @torch.no_grad()
    def _predict(self, images, intrinsics, extrinsics):
        """Predict cameras and undo image padding/resize on their intrinsics.

        Camera conditions use BF16, with intrinsics quantized in normalized
        coordinates before pixel scaling. Return CPU matrices in the input
        canvas convention and offload the model after success or failure.
        """
        inputs, transforms = prepare_mapanything_views(images, intrinsics, extrinsics)
        # Quantize normalized calibration before converting to processed pixels.
        # normalize to the processed canvas, cast to bf16, then scale to pixels.
        # Rounding only after pixel scaling changes the 518px anchor conditions.
        for view in inputs:
            if "intrinsics" in view:
                height, width = view["img"].shape[-2:]
                matrix = view["intrinsics"].clone()
                matrix[:, 0] /= width
                matrix[:, 1] /= height
                matrix = matrix.to(device=self.device, dtype=torch.bfloat16)
                matrix[:, 0] *= width
                matrix[:, 1] *= height
                view["intrinsics"] = matrix
            if "camera_poses" in view:
                view["camera_poses"] = view["camera_poses"].to(device=self.device, dtype=torch.bfloat16)
        cache_hooks = []
        predictions = None
        try:
            self.model.to(self.device)
            if self.low_vram and self.device.type == "cuda":
                # Geometric fusion allocates large FP32 temporaries. Keeping
                # their unused allocator blocks can spill later attention into
                # WDDM shared memory even when live tensors fit dedicated VRAM.
                cache_hooks.append(self.model.info_sharing.register_forward_pre_hook(
                    lambda module, args: torch.cuda.empty_cache()))
                cache_hooks.append(self.model.info_sharing.register_forward_hook(
                    lambda module, args, output: torch.cuda.empty_cache()))
            predictions = self.model.infer(
                inputs, memory_efficient_inference=True, minibatch_size=self.minibatch_size,
                use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=True,
                apply_confidence_mask=False, confidence_percentile=10, use_multiview_confidence=False,
                ignore_calibration_inputs=False, ignore_depth_inputs=False, ignore_pose_inputs=False,
                ignore_depth_scale_inputs=False, ignore_pose_scale_inputs=False,
            )
            if len(predictions) != len(images):
                raise ValueError("MapAnything did not return a camera for every image")
            # Convert poses to world-to-camera before projecting rotations to SO(3).
            exts = torch.stack([torch.linalg.inv(p["camera_poses"]).cpu() for p in predictions]).squeeze(1)
            exts = torch.from_numpy(remove_rotation_scale(exts.numpy()))
            calibrations = []
            for image, predicted, transform in zip(images, predictions, transforms):
                matrix = np.linalg.solve(transform, predicted["intrinsics"][0].float().cpu().numpy())
                matrix[0] /= image.width
                matrix[1] /= image.height
                calibrations.append(matrix)
            return exts, torch.from_numpy(np.stack(calibrations)).float()
        finally:
            for hook in cache_hooks:
                hook.remove()
            if self.low_vram:
                self.model.cpu()
                predictions = None
                inputs = None
                if self.device.type == "cuda":
                    torch.clear_autocast_cache()
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def __call__(self, scene, images, intrinsics, extrinsics):
        """Fill only absent fields; known cameras remain model conditions.

        Return lists of normalized intrinsics and canonical OpenCV extrinsics.
        Anchor cameras provide metric alignment after prediction; their poses
        are not supplied as model conditions. last_report stores round residuals.
        """
        if not images or len(images) != len(intrinsics) or len(images) != len(extrinsics):
            raise ValueError("Images and camera lists must have matching nonzero lengths")
        self.last_report = []
        known_intrinsics = [None if x is None else np.asarray(x).copy() for x in intrinsics]
        known_extrinsics = [None if x is None else np.asarray(x).copy() for x in extrinsics]
        if all(i is not None and e is not None for i, e in zip(intrinsics, extrinsics)):
            return known_intrinsics, known_extrinsics
        previous = []
        count = len(images)
        # Repeat anchor rendering, joint prediction and frame alignment twice.
        for round_index in range(2):
            anchors = self._anchors(previous)
            print(f"Estimating reference cameras: round {round_index + 1}/2, {len(anchors)} anchors", flush=True)
            anchor_images = self._render_anchors(scene, anchors)
            print(f"Predicting reference cameras from {count + len(anchors)} views", flush=True)
            # Reference images come first; predicted anchor centers set the mesh frame.
            predicted_ext, predicted_intr = self._predict(
                images + anchor_images,
                known_intrinsics + [intr.cpu().numpy() for _, intr in anchors],
                known_extrinsics + [None] * len(anchors),
            )
            true_centers = torch.stack([torch.linalg.inv(ext)[:3, 3] for ext, _ in anchors]).double().cpu().numpy()
            predicted_centers = torch.linalg.inv(predicted_ext[count:])[:, :3, 3].double().numpy()
            similarity, inliers, residuals = align_camera_centers(
                predicted_centers, true_centers, sample_size=len(anchors) // 2, threshold=.05 * self.anchor_radius,
            )
            aligned = align_extrinsics(predicted_ext, torch.from_numpy(similarity).float())
            result_ext = [known if known is not None else aligned[i].numpy().copy() for i, known in enumerate(known_extrinsics)]
            # Limit estimated focal lengths to frame a radius-0.7 canonical object.
            for index in range(count):
                if known_intrinsics[index] is not None:
                    continue
                pose = torch.linalg.inv(torch.as_tensor(result_ext[index], dtype=torch.float32))
                cap = .5 * torch.linalg.norm(pose[:3, 3]) / .7
                side = max(images[index].size)
                predicted_intr[index, 0, 0] = min(cap * side / images[index].width, predicted_intr[index, 0, 0])
                predicted_intr[index, 1, 1] = min(cap * side / images[index].height, predicted_intr[index, 1, 1])
            result_intr = [known if known is not None else predicted_intr[i].numpy().copy() for i, known in enumerate(known_intrinsics)]
            previous = [torch.from_numpy(ext).float() for ext in result_ext]
            self.last_report.append({"anchors": len(anchors), "inliers": int(inliers.sum()),
                                     "anchor_rms": float(np.sqrt(np.mean(residuals[inliers]**2)))})
        return result_intr, result_ext
