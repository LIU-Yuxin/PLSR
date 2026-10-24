# PLSR 模型仓库准备

开发仓库中的 `model_repo/` 是未来 Hugging Face repo 的模板。上传该目录中的内容，
再补上两份 adapter；无需上传 TRELLIS.2 基础权重。

```text
PLSR/
├── README.md
├── pipeline.json
├── shape/
│   ├── config.json
│   └── shape_adapter.safetensors
└── pbr/
    ├── config.json
    └── pbr_adapter.safetensors
```

`pipeline.json` 指定两个模型的相对配置/adapter 路径、对应的 TRELLIS.2
基础权重路径，并在 `inference_defaults` 中保存默认分辨率序列、Shape/PBR
采样参数、seed 和输出方式。该块不包含图片、mesh 或输出目录。
`PLSRPipeline.from_pretrained` 同时加载模型与这些默认值；调用 `run` 时只需
提供自己的输入和输出位置，可以逐项覆盖默认参数。
DINOv3、MapAnything、RMBG 的权重与配置不放入此 repo。
Shape/PBR 的 `config.json` 保存实际模型结构、LoRA、normalization 和
sigma_min。模型文件中不需要哈希、大小或训练步数清单。

两份 adapter 使用 `safetensors.torch.save_file(state_dict, path)` 保存。
保持训练时保存的参数名，包含 LoRA 与 PLSR 新增层，不要包在
`{"state_dict": ...}` 中，也不要加入优化器等训练状态。现有的
`shape_denoiser_step0030000.pt` / `pbr_denoiser_step0015000.pt` 保存的是纯 tensor 字典，
需要转换并另存为上述 `.safetensors` 文件，不必重新训练或合并 LoRA。

维护者可以在开发仓库、已安装 PyTorch 的环境中准备一个新的上传目录：

```bash
python scripts/prepare_hf_repo.py /path/to/upload \
  --shape-weights /path/to/shape_adapter.pt \
  --pbr-weights /path/to/pbr_adapter.pt
```

该工具将 `.pt` 转换为 `.safetensors`；若源文件已是 `.safetensors`，则直接复制，
另存到上传目录。它不更改源权重，也不上传任何文件。

主入口 `plsr/__main__.py` 默认使用 `yxliu/PLSR` 和各公共组件的官方 repo ID。
`PLSR_REPO_ID`、`TRELLIS2_REPO_ID`、`DINO_REPO_ID`、`MAPANYTHING_REPO_ID`、
`RMBG_REPO_ID` 均接受同名环境变量覆盖。用户下载整个 repo 后，可在运行时
把相应环境变量设为下载目录，无需在发布代码中填写本地路径。
公共组件从其独立 repo 加载，不随 PLSR 权重发布；运行入口不接受单独的权重路径。

Python API 可以单独指定 DINO 来源；它同时支持 HF repo ID 和包含
`config.json`、模型 `.safetensors` 的 HF 格式目录：

```python
from plsr import PLSRPipeline

if __name__ == "__main__":
    pipeline = PLSRPipeline.from_pretrained(
        "yxliu/PLSR", trellis2_repo_id="microsoft/TRELLIS.2-4B",
        dino_repo_id="/path/to/dinov3-hf-repo",
    )
    result = pipeline.run(
        images=["/path/to/reference.png"], mesh="/path/to/coarse.glb",
        output_dir="/path/to/result",
    )
```

加载顺序为：按配置构造 PLSR 模型 → 载入 TRELLIS.2 基础参数 → 装配
LoRA/SparseTensor 适配 → 载入 PLSR adapter。模型保持 FP16 torso 和未合并
LoRA 的既有执行方式，固定 gate/norm 参数沿用模型构造时的默认值。
