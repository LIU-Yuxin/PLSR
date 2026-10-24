"""Run small CUDA operations for the native dependencies used by inference."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="Also check camera-estimation lighting")
    args = parser.parse_args()
    import torch

    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": None,
        "gpu_memory_gib": None,
        "checks": {},
    }
    if not report["cuda_available"]:
        print(json.dumps(report, indent=2))
        return 1
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    report["gpu"] = properties.name
    report["gpu_memory_gib"] = round(properties.total_memory / 2**30, 2)

    def check(name, operation):
        try:
            operation()
            torch.cuda.synchronize()
            report["checks"][name] = "ok"
        except Exception as exc:
            report["checks"][name] = f"{type(exc).__name__}: {exc}"

    def torch_check():
        assert torch.isfinite(torch.ones(1, device=device)).all()

    def flash_check():
        from flash_attn import flash_attn_func

        q = torch.randn(1, 128, 4, 64, device=device, dtype=torch.float16)
        out = flash_attn_func(q, q, q)
        assert out.shape == q.shape and torch.isfinite(out).all()

    def xformers_check():
        from xformers.ops import memory_efficient_attention

        q = torch.randn(1, 128, 4, 64, device=device, dtype=torch.float16)
        out = memory_efficient_attention(q, q, q)
        assert out.shape == q.shape and torch.isfinite(out).all()

    def sdpa_check():
        from torch.nn.functional import scaled_dot_product_attention

        q = torch.randn(1, 4, 128, 64, device=device, dtype=torch.float16)
        out = scaled_dot_product_attention(q, q, q)
        assert out.shape == q.shape and torch.isfinite(out).all()

    dense_attention = os.environ.get("ATTN_BACKEND", "flash_attn")
    sparse_attention = os.environ.get("SPARSE_ATTN_BACKEND", dense_attention)
    if dense_attention not in {"flash_attn", "flash_attn_3", "xformers", "sdpa", "naive"}:
        dense_attention = "flash_attn"
    if sparse_attention not in {"flash_attn", "flash_attn_3", "xformers", "sdpa"}:
        sparse_attention = "flash_attn"
    report["attention_backends"] = {"dense": dense_attention, "sparse": sparse_attention}
    attention_checks = []
    if "xformers" in (dense_attention, sparse_attention):
        attention_checks.append(("xformers", xformers_check))
    if "sdpa" in (dense_attention, sparse_attention):
        attention_checks.append(("sdpa", sdpa_check))
    if any(backend.startswith("flash_attn") for backend in (dense_attention, sparse_attention)):
        attention_checks.append(("flash_attn", flash_check))

    def pytorch3d_check():
        from pytorch3d.ops import knn_points

        points = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device=device)
        distances, indices, _ = knn_points(points, points, K=1)
        assert torch.equal(indices.squeeze(-1), torch.tensor([[0, 1]], device=device))
        assert torch.allclose(distances, torch.zeros_like(distances))

    def flexgemm_check():
        from flex_gemm.kernels import cuda

        coords = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [0, 3, 2, 1], [0, 7, 7, 7]],
                              device=device, dtype=torch.int32)
        values = torch.tensor([7, 11, 13, 17], device=device, dtype=torch.uint32)
        keys = torch.full((32,), torch.iinfo(torch.uint32).max, device=device, dtype=torch.uint32)
        slots = torch.empty((32,), device=device, dtype=torch.uint32)
        cuda.hashmap_insert_3d_cuda(keys, slots, coords, values, 8, 8, 8)
        result = cuda.hashmap_lookup_3d_cuda(keys, slots, coords, 8, 8, 8)
        assert torch.equal(result, values)

    def diffoctreerast_check():
        from diffoctreerast import _C

        assert _C is not None

    def nvdiffrast_check():
        import nvdiffrast.torch as dr

        context = dr.RasterizeCudaContext()
        positions = torch.tensor([[[-0.5, -0.5, 0, 1], [0.5, -0.5, 0, 1], [0, 0.5, 0, 1]]],
                                 device=device, dtype=torch.float32)
        triangles = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
        raster, _ = dr.rasterize(context, positions, triangles, resolution=[16, 16])
        assert (raster[..., 3] > 0).any()

    def cumesh_check():
        import cumesh

        assert cumesh.CuMesh() is not None

    def ovoxel_check():
        from o_voxel import _C

        assert _C is not None

    def nvdiffrec_check():
        from nvdiffrec_render.light import EnvironmentLight

        light = EnvironmentLight(torch.full((6, 64, 64, 3), 0.5, device=device))
        light.build_mips()
        assert torch.isfinite(light.diffuse).all()

    checks = [
        ("torch", torch_check),
        *attention_checks,
        ("pytorch3d", pytorch3d_check),
        ("flex_gemm", flexgemm_check),
        ("diffoctreerast", diffoctreerast_check),
        ("nvdiffrast", nvdiffrast_check),
        ("cumesh", cumesh_check),
        ("o_voxel", ovoxel_check),
    ]
    if args.camera:
        checks.append(("nvdiffrec_render", nvdiffrec_check))
    for name, operation in checks:
        check(name, operation)
    print(json.dumps(report, indent=2))
    return int(any(value != "ok" for value in report["checks"].values()))


if __name__ == "__main__":
    sys.exit(main())
