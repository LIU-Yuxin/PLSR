
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, IS_HIP_EXTENSION
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")  # 'auto' | 'cuda' | 'rocm'
DEBUG_SANITIZE = os.environ.get("SANITIZE", "0") in ("1", "true", "True")
DEVICE_DEBUG = os.environ.get("CUDA_DEVICE_DEBUG", "0") in ("1", "true", "True")  # enables -G

# Detect HIP vs CUDA
if BUILD_TARGET == "auto":
    IS_HIP = bool(IS_HIP_EXTENSION)
elif BUILD_TARGET == "cuda":
    IS_HIP = False
elif BUILD_TARGET == "rocm":
    IS_HIP = True
else:
    raise RuntimeError(f"Unknown BUILD_TARGET={BUILD_TARGET}")

# Compute arch flags
if not IS_HIP:
    # CUDA: cc_flag is empty unless you want to force archs
    cc_flag = []
else:
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    cc_flag = [f"--offload-arch={arch}" for arch in archs]

# Base flags
cxx_common = (["/std:c++17", "/O2", "/Zc:preprocessor", "/DUSE_CUDA"]
              if os.name == "nt" else ["-std=c++17", "-O3"])
nvcc_common = (["-std=c++17", "-O3", "-DUSE_CUDA", "-Xcompiler=/Zc:preprocessor"]
               if os.name == "nt" else ["-std=c++17", "-O3"]) + cc_flag

# Debug/sanitize tuning
asan_ubsan_cxx = ["-g", "-O1", "-fno-omit-frame-pointer", "-fsanitize=address", "-fsanitize=undefined"]
asan_ubsan_link = ["-fsanitize=address", "-fsanitize=undefined"]

# Device-side debug (CUDA only): -G disables many optimizations on device code
nvcc_device_debug = ["-G", "-lineinfo"] if DEVICE_DEBUG and not IS_HIP else ["-lineinfo"]

# Apply debug flags
if DEBUG_SANITIZE:
    cxx_flags = cxx_common + asan_ubsan_cxx
    if not IS_HIP:
        # CUDA: host sanitizer flags must go through -Xcompiler
        nvcc_flags = nvcc_common + nvcc_device_debug + [
            "-Xcompiler", "-g",
            "-Xcompiler", "-O1",
            "-Xcompiler", "-fno-omit-frame-pointer",
            "-Xcompiler", "-fsanitize=address",
            "-Xcompiler", "-fsanitize=undefined",
        ]
        link_flags = asan_ubsan_link
    else:
        # HIP/ROCm (clang-based): pass sanitizer flags directly
        nvcc_flags = nvcc_common + nvcc_device_debug + asan_ubsan_cxx
        link_flags = asan_ubsan_link
else:
    cxx_flags = cxx_common
    nvcc_flags = nvcc_common + ["-lineinfo"]
    link_flags = []

setup(
    name="o_voxel",
    packages=[
        "o_voxel",
        "o_voxel.convert",
        "o_voxel.io",
    ],
    ext_modules=[
        CUDAExtension(
            name="o_voxel._C",
            sources=[
                # Hashmap functions
                "src/hash/hash.cu",
                # Convert functions
                "src/convert/flexible_dual_grid.cpp",
                "src/convert/volumetic_attr.cpp",
                # Serialization functions
                "src/serialize/api.cu",
                "src/serialize/hilbert.cu",
                "src/serialize/z_order.cu",
                # IO functions
                "src/io/svo.cpp",
                "src/io/filter_parent.cpp",
                "src/io/filter_neighbor.cpp",
                # Rasterization functions
                "src/rasterize/rasterize.cu",
                # main
                "src/ext.cpp",
            ],
            include_dirs=[
                os.path.join(ROOT, "third_party/eigen"),
            ],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
            extra_link_args=link_flags,  # <-- ensure libasan/libubsan are linked
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
