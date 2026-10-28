# Venice mask examples

The coarse mesh was encoded and decoded with the TRELLIS.2 Shape/PBR VAEs at
1024 voxel resolution, then exported with PBR textures. The six 2048 x 2048 RGBA
reference images were rendered from the original model. No PLSR refinement was
used to prepare the coarse mesh.

From the repository root, after installation and model access are configured:

```bash
python -m plsr infer assets/venice_mask/known_cameras.json --low-memory
```

This encodes the supplied coarse GLB at 640 and refines it to 1280. The result is
`outputs/venice_mask/mesh_1280.glb`. Use
`--output-dir /path/to/result` to choose another location. Add
`--resolutions 640 1280 2560` to continue to 2560.

To estimate the six cameras from the same images and coarse mesh, run:

```bash
python -m plsr infer assets/venice_mask/estimated_cameras.json --low-memory
```

This configuration omits all camera parameters and writes to
`outputs/venice_mask_estimated_cameras`. It requires the camera-estimation
dependencies and models. Both examples save silhouette overlays in
`view_diagnostics/` and scores in `view_diagnostics.json` inside their output
folder. `--min-view-iou 0.6` controls the minimum accepted overlap; views below
it are discarded, and inference stops if no usable view remains.

Estimated-camera views are automatically centered before silhouette alignment.
Add `--independent-camera-estimation` to estimate each image separately against
rendered mesh views. The JSON option `independent_camera_estimation` defaults to
`false`; `--no-independent-camera-estimation` overrides it back to joint estimation.

`known_cameras.json` contains the image paths and camera parameters. Intrinsics are
3 x 3 matrices in pixels for the original 2048 x 2048 images. Extrinsics are
4 x 4 OpenCV world-to-camera transforms in the coarse GLB's coordinate frame
(x right, y down, z forward in camera space). The coarse mesh retains the
original model's world coordinates. Complete cameras and image alpha skip
camera estimation, image alignment and background removal.

Source: [Venice Mask by DailyArt](https://sketchfab.com/3d-models/venice-mask-4aace12762ee44cf97d934a6ced12e65),
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
The coarse GLB and rendered images are derived from this model; this asset
license is separate from the code license.
