# PLSR 推理使用

使用 Python API 或 `python -m plsr infer`：按输入装配模型，支持已有 mesh / 单图生成 coarse 后渐进推理、缺失相机估计及二维轮廓匹配、逐级 latent 保存和局部解码 GLB 输出。

## 统一推理入口

`PLSRPipeline` 负责完整任务：准备输入、编码基础 mesh、逐级细化并导出 GLB。
其内部的 `PLSRRefinementStep` 只执行一次分辨率翻倍，接收和返回 Shape/PBR
latent；同一个 step 在多级循环中重复使用已加载的模型。

安装步骤和依赖版本见 [INSTALL.md](INSTALL.md)。默认模型来源为 `yxliu/PLSR`
和 `microsoft/TRELLIS.2-4B`。主入口 `plsr/__main__.py` 中的默认 repo ID
可以通过同名环境变量覆盖，值可以是 repo ID 或已下载的 HF repo 根目录。
图片、可选 mesh 和输出目录在运行时显式提供，无需创建配置文件：

```bash
python -m plsr infer --image /path/to/reference.png --mesh /path/to/coarse.glb --output-dir /path/to/result
# 不提供 mesh 时，先从单张图片生成粗模型。
python -m plsr infer --image /path/to/reference.png --output-dir /path/to/result
```

入口通过 `PLSRPipeline.from_pretrained` 装配模型：加载 TRELLIS.2 基础参数，
再装配 LoRA 并加载 PLSR 的 `shape_adapter.safetensors` / `pbr_adapter.safetensors`。下载和缓存交给 Hugging Face；
不需要 `models.json`、本地 `ckpts` 文件布局或权重哈希清单。用户自行下载整个
模型 repo 后，可通过对应环境变量指定下载目录，文件结构保持一致。

默认推理参数来自 PLSR 模型 repo 的 `pipeline.json` 中的 `inference_defaults`。
实际分辨率、Shape/PBR 采样参数、seed 和输出方式以模型 repo 为准。
可用 `--resolutions`、`--shape-guidance`、`--pbr-guidance`、
`--steps`、`--seed`、`--decode-intermediate`、`--output-space` 覆盖。
模型 repo 不保存用户输入或输出路径，也不要求固定的 `inputs` / `outputs` 文件夹。

模型来源环境变量为 `PLSR_REPO_ID`、`TRELLIS2_REPO_ID`、`DINO_REPO_ID`、
`MAPANYTHING_REPO_ID`、`RMBG_REPO_ID`。DINOv3、MapAnything、RMBG 默认从各自
公共 repo 加载，不包含在 PLSR 模型 repo 中。本地 DINOv3 可单次指定：

```bash
DINO_REPO_ID=/path/to/dinov3-hf-repo python -m plsr infer --image /path/to/reference.png --output-dir /path/to/result
```

目录应包含原 repo 的 `config.json` 和权重。来源优先级为显式 Python API 参数
→ 环境变量 → 主入口中的默认 repo ID；环境变量在进程启动时读取，无需修改源码。

仍需准备输入时，可选阶段按原始输入选择；已有 `coarse.glb` 时复用它：

- 没有 mesh：加载原始 TRELLIS.2 coarse 模型和其配置指定的 sparse-structure decoder。
- 任何参考图缺相机字段：加载 MapAnything；全已知时跳过相机估计和二维轮廓匹配。
- 任意 RGB 或不透明 RGBA 图片：加载 RMBG；所有图片都有有效 alpha 时跳过去背景。

MapAnything 还需要已安装的 MapAnything / UniCeption 包，其 DINOv2 由 Torch Hub
缓存。首次获取模型需要网络及对应仓库的访问权限；离线运行可预先准备 HF /
Torch Hub 缓存。`HF_HOME` 可以改变 HF 缓存目录。

CLI 默认每 part 简化上限 500000、纹理尺寸 4096、Shape/PBR patch batch 12/8、CPU threads 4。简化目标为 `min(上限, max(1, face_count // 5))`；每个 part 使用相同的纹理尺寸。每块使用 `remesh=True, remesh_band=1, remesh_project=0`，再组装全部重叠 part。可用 `--texture-size` 显式统一纹理尺寸，或用 `--decimation-target`、`--shape-batch-size`、`--pbr-batch-size`、`--cpu-threads` 覆盖对应设置。CLI 以非零状态报告失败，具体进度和错误保留在输出中。中断后直接重跑同一命令，自动复用输出目录中的 latent 和 GLB。

显存受限时，先尝试 `--low-memory`：它设置 Shape/PBR batch 为 4/4、相机视角 batch 为 1、相机 MLP token batch 为 2048，并启用相机骨干和中间特征的 CPU 卸载。分辨率、采样步数、patch、解码和纹理设置均保留。它是可显式选择的运行预设，不按显卡型号自动启用，也不保证任意资产都适合某个显存容量。已有相机内外参时会跳过相机估计。

```bash
python -m plsr infer --image /path/to/reference.png --output-dir /path/to/result --low-memory
```

可用 `--shape-batch-size`、`--pbr-batch-size`、`--camera-batch-size`、`--camera-token-batch-size` 覆盖预设；`--no-camera-offload-encoder` 和 `--no-camera-offload-features` 可单独关闭卸载。全局相机注意力始终保留所有视角。Python API 使用 `RunOptions.for_low_memory(shape_batch_size=2)`，未传入的参数继承该预设。分批可能产生浮点舍入差异；CPU 卸载增加传输和系统内存开销。

Windows 上还应观察专用和共享 GPU 内存：使用大量共享内存可能表现为严重减速，而不出现 OOM。分别观察相机估计、SR 采样和解码烘焙阶段；Shape/PBR batch 无法控制相机阶段的内存。

若仍需降低资源开销，可用 `--resolutions 128 256 512` 降低分辨率，再将 Shape/PBR batch 降为 1。`--sampling-patch-size` 控制两个 DiT 的局部 SLat patch 边长，每 patch 的稀疏 token 数不超过边长的三次方；配套的 `--sampling-patch-overlap` 必须小于边长。最终 VAE 解码可调 `--decode-local-resolution`、`--decode-patch-size`、`--decode-overlap` 和 `--decode-cell-limit`，后一项限制每个解码块的 latent cells。默认仍分别是 1024、64、2、25000。示例：

```bash
python -m plsr infer --image /path/to/reference.png --mesh /path/to/coarse.glb --output-dir /path/to/result \
  --resolutions 128 256 512 \
  --shape-batch-size 1 --pbr-batch-size 1 \
  --sampling-patch-size 16 --sampling-patch-overlap 4 \
  --decode-local-resolution 512 --decode-patch-size 32 \
  --decode-overlap 2 --decode-cell-limit 4000 --texture-size 1024
```

这些覆盖值会记录在 `runtime.json` 和 `report.json`，作用于本次尚未完成的计算。已有产物直接复用；更改参数后需要重新计算时，使用新的输出目录。减小采样 patch 会改变局部条件和融合结果；减小解码块可能增加接缝和少面数后处理失败的风险。基础 mesh 的体素化和 Shape/PBR VAE **编码**、每级的低分辨率特征解码，以及单图模式的原始 TRELLIS.2 coarse 生成目前仍以各自的完整工作单元执行；降低 batch 或最终解码块并不能保证它们适合小显存 GPU。真正的编码切块需要确定重叠上下文并验证边界输出，尚未作为安全开关提供。

入口在导入 Torch 前为未配置的环境项设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`OPENCV_IO_ENABLE_OPENEXR=1`。用户已设置的值保留，并写入 `runtime.json`。Python 调用建议使用新进程，确保这些设置早于 Torch/CUDA 初始化。

```python
from plsr import RunOptions, run_inference
from plsr.__main__ import PLSR_REPO_ID, TRELLIS2_REPO_ID

def main():
    result = run_inference(
        {"images": [{"path": "/path/to/reference.png"}],
         "mesh": "/path/to/coarse.glb", "output_dir": "/path/to/result"},
        PLSR_REPO_ID,
        trellis2_repo_id=TRELLIS2_REPO_ID,
        options=RunOptions(),
    )
    print(result.mesh_path)

if __name__ == "__main__":
    main()
```

`__main__` 保护用于逐 part 烘焙的独立进程。入口加载核心模型后，pipeline 根据已有文件选择尚需执行的阶段；仅在需要准备源 mesh 或参考条件时读取和检查输入。`runtime.json` 保存本次已装配的阶段、模型 repo 来源、VAE dtype、运行选项及环境，`report.json` 记录本次执行状态，并以 `reused` 标记复用的层。每级 Shape/PBR latent 保存后可供下一次调用继续使用；已有最终 latent 和坐标变换时，只补缺失的解码导出，跳过相机和图片条件准备。续跑从已保存的层开始，不从单个采样步或烘焙 part 内继续。

## 输入检查

输入 mesh 使用 glTF PBR 材质，支持纹理或常量材质，并保留场景实例变换和逐面材质分配。顶点颜色和其他材质类型目前不支持。纹理无需预先整理成正方形：体素化前会将各材质纹理按长边最近的 2 次幂统一缩放为正方形，UV 坐标保持不变；已符合要求的纹理保留原值。


在仓库根目录执行：

```bash
python -m plsr validate --image /path/to/reference.png --mesh /path/to/coarse.glb --output-dir /path/to/result
```

该命令读取模型 repo 的默认参数、检查 mesh 文件可读且非空、完整解码参考图片。
它输出 JSON 检查结果，不创建输出目录、不加载权重、不使用 GPU。
首次读取在线 repo 的 `pipeline.json` 需要网络；本地 repo 直接读取。
`inputs_valid` 不表示 mesh 几何、模型权重或完整推理已验证。

图片原始短边小于 1024 像素时，JSON 中的 `warnings` 会指出图片、尺寸和细节不足的影响；检查仍然成功。不会插值放大图片来绕过警告。

需要保存实验请求或提供相机矩阵时，可以自行创建一个可选的请求 JSON。
它只需包含任务输入；推理参数继续继承模型 repo 的默认值：

```json
{
  "mesh": "/path/to/coarse.glb",
  "images": [{"path": "/path/to/reference.png"}],
  "output_dir": "/path/to/result"
}
```

通过 `python -m plsr infer /path/to/request.json` 读取自己的请求。
用户自行提供相对路径时，CLI/Python 直接传入的路径以当前工作目录为基准；
JSON 内的路径以该 JSON 所在目录为基准。支持 `~`，不执行 shell 展开。
请求中的参数覆盖 repo 默认值；只覆盖某个 sampler 的 guidance 时，其余步数、
interval 等设置仍继承 repo，不必复制整组参数。

序列至少包含基础编码分辨率和一个 SR 目标。首项为正的 128 整数倍，每项严格翻倍；SR 对应后续各项。

省略 `mesh` 或设为 `null` 表示单图生成 coarse 的输入模式。此模式恰好接受一张图片，公共 `run_inference` / CLI 自动装配 coarse 阶段；手动构造底层 `PLSRPipeline` 时需提供 `coarse_generator`，通过 `from_pretrained` 创建时自动处理。已有 mesh 时可提供多张参考图。

## 可选相机字段

每个 image 对象可分别提供 `intrinsics` 和 `extrinsics`，缺失项保持为未知，检查阶段不会估计或覆盖相机：

- `intrinsics`：3×3 无 skew 的针孔内参，焦距为正，底行为 `[0,0,1]`。
- `intrinsics_units`：`normalized`（默认，第一行除以原图宽、第二行除以原图高）或 `pixels`（原图像素单位）。
- `extrinsics`：4×4 **输入 mesh 世界坐标到相机坐标**变换；使用 OpenCV 相机坐标约定：x 向右、y 向下、z 向前。包含刚体旋转和平移，底行为 `[0,0,0,1]`。

这些字段描述原始图片和输入 mesh，尚未进行 canonicalization 或图像坐标转换。字段校验保留已提供的值；后续 pipeline 会根据约定转换，并只估计缺失字段。

## Python API

```python
from plsr import RefinementConfig
from plsr.inputs import validate_inputs
from plsr.pretrained import load_inference_defaults
from plsr.__main__ import PLSR_REPO_ID

config = RefinementConfig.from_json(
    "/path/to/request.json", defaults=load_inference_defaults(PLSR_REPO_ID),
)
report = validate_inputs(config)
print(report.input_mode, report.needs_camera_estimation)
```

API 使用标准 Python warnings 发出 `LowResolutionImageWarning`；`validate` 将警告列在 JSON 报告中，`infer` 将警告写入 stderr。配置中的未知字段会报错，避免参数拼写错误被忽略。

## 直接创建 pipeline

```python
from plsr import PLSRPipeline

if __name__ == "__main__":
    pipeline = PLSRPipeline.from_pretrained()
    result = pipeline.run(
        images=["/path/to/reference.png"],
        mesh="/path/to/coarse.glb",  # 省略时先生成粗模型。
        output_dir="/path/to/result",
    )
```

`from_pretrained` 加载核心组件及 `pipeline.inference_defaults`；`run` 按请求加载可选阶段。
可以直接传入 `seed=123`、`shape_sampler={"guidance_strength": 4.0}` 等覆盖设置。
需要先构造请求时，可调用 `pipeline.make_config(images=..., output_dir=..., mesh=...)`；
现有的 `pipeline.run(RefinementConfig(...))` 仍接受完整、显式的请求。
普通 CLI 使用前面的
`run_inference`，它负责在模型导入前配置运行环境并记录运行选项。

## 图片和相机处理

原图短边小于 1024 时发出警告并继续。已有有效 alpha 的图片保留其透明度，包括 RGBA、灰度加 alpha 和调色板 PNG；RGB 或全不透明图片通过 RMBG 获得前景 alpha。参考图按长边缩放后居中补边到公共画布，不裁掉前景，内参同步使用实际缩放比例和补边偏移。

缺失相机字段使用 MapAnything 估计，已提供的内参或外参保留并参与条件。估计器将输入补成方形后按 MapAnything 的 518 像素变换预处理，预测后把内参还原到参考画布；两轮渲染锚点和相似变换拟合把相机放到标准化 mesh 坐标中。

带估计字段的参考图再按渲染轮廓执行二维位置/尺度匹配，匹配只移动图像，不再次修改相机。全已知相机跳过估计与匹配。轮廓使用第一层表面的材质 alpha；该处理不能修复视角差异或粗模几何错误。对齐路径保留两阶段软 alpha 合成，半透明边缘相较全已知相机路径会进一步衰减。

## 输出与运行状态

输出目录保存中间结果和最终模型，重跑时自动复用。主要文件包括：

- `mesh_<最终分辨率>.glb`：包含全部解码part的完整物体，保留重叠表面。
- `latents_<分辨率>.npz`：每级对齐的Shape/PBR latent。
- `transforms.json`、`cameras.json`：坐标变换及本次使用的相机。
- `config.json`、`runtime.json`、`report.json`：请求、执行设置与完成/失败状态。
- `timings.jsonl`：后台逐阶段耗时，包括核心模型加载、粗模型生成/导出、相机和参考图准备、基础编码、每级 Shape/PBR 采样、latent 写盘、最终解码导出与总耗时；失败阶段也写入状态。CUDA 阶段同时记录该阶段的峰值 allocated/reserved 显存字节数，便于定位小显存瓶颈。不会显示在生成 UI 中。
- `decode_events.jsonl`：逐行持久化粗模型和最终模型的分块解码事件，包括每块选中的 latent cell 数与范围、VAE 解码体素数、提取面数、烘焙重试和合并进度。失败时从最后一条事件定位具体阶段和 part 编号；`report.json` 的 `decode_events` 字段指向此文件。若还未进入解码阶段，该文件尚不存在。
- 单图模式额外保存`coarse.glb`、`coarse_latents.npz`和coarse设置；coarse是内部中间结果。

`output_space="input"`默认恢复输入mesh坐标；单图模式对应生成的coarse坐标。设为`canonical`可保留标准化坐标。`decode_intermediate=true`可额外导出中间级预览。

直接重跑相同的输出目录即可继续：所需 GLB 已齐全时直接返回；已有最终 latent 时补解码导出；否则从最近的已保存 latent 继续细化。中间级预览只补缺失的 GLB。续跑根据产物文件判断，不依赖旧 `config.json`、`runtime.json` 或 `report.json`。这些记录更新为本次请求；更改输入或参数后要重新计算时，使用新输出目录，或删除需要重算的产物。

解码导出会显示总 part 数、当前 part 的 Shape/PBR 解码与重网格化/纹理烘焙阶段，以及最终 GLB 写入。采样结束后仍需要完成这些步骤。面数上限和纹理尺寸按每个 part 应用，高分辨率输出可能较大；已有的 `--texture-size`、`--decimation-target` 可降低导出体量。`--decode-intermediate` 导出中间 SR 层，不包含基础编码层；原始 TRELLIS coarse mesh 与基础编码重建的分辨率不同。

新环境验证步骤见 [VALIDATION.md](VALIDATION.md)。
