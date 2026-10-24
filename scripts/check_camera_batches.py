"""Compare camera batching with upstream blocks, without downloading weights."""

import argparse
from contextlib import redirect_stdout
import copy
import json
from pathlib import Path
import sys


def check_batches(device="cpu"):
    import torch
    from torch import nn
    from uniception.models.utils.transformer_blocks import SelfAttentionBlock, SwiGLUFFN

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from trellis2.pipelines.camera_batches import configure_mapanything_batches, _ViewBatchedBlock

    torch.manual_seed(42)
    torch.set_num_threads(4)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; use --device cpu for the CPU comparison")

    class PositionOffset(nn.Module):
        def forward(self, values, positions):
            return values + positions[..., 0].unsqueeze(1).unsqueeze(-1) * .001

    class EncoderModel(nn.Module):
        def __init__(self, registers=True):
            super().__init__()
            self.encoder = nn.Conv2d(3, 32, 1)
            self.registers = registers
            self.info_sharing = nn.Module()
            self.info_sharing.self_attention_blocks = nn.ModuleList()

        def _encode_n_views(self, views):
            encoded = self.encoder(torch.cat([view["img"] for view in views]))
            registers = encoded.flatten(2)[..., :2].chunk(len(views)) if self.registers else None
            return encoded.chunk(len(views)), registers

    comparisons = []
    dtypes = [torch.float32]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    for mlp in (None, SwiGLUFFN):
        model = EncoderModel().eval().to(device)
        kwargs = {} if mlp is None else {"mlp_layer": mlp}
        model.info_sharing.self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock(32, num_heads=4, qk_norm=True,
                               custom_positional_encoding=PositionOffset(), **kwargs)
            for _ in range(2)
        ]).eval().to(device)
        reference = copy.deepcopy(model)
        keys = set(model.state_dict())
        storage = sorted(parameter.data_ptr() for parameter in model.parameters())
        configure_mapanything_batches(model, 2, 11)
        assert not isinstance(model.info_sharing.self_attention_blocks[0], _ViewBatchedBlock)
        assert isinstance(model.info_sharing.self_attention_blocks[1], _ViewBatchedBlock)
        assert storage == sorted(parameter.data_ptr() for parameter in model.parameters())
        for dtype in dtypes:
            for index, shape in ((0, (1, 65, 32)), (1, (5, 17, 32))):
                inputs = torch.randn(shape, device=device)
                positions = torch.randn((*shape[:2], 2), device=device)
                with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                     enabled=dtype == torch.bfloat16):
                    expected = reference.info_sharing.self_attention_blocks[index](inputs, positions)
                    actual = model.info_sharing.self_attention_blocks[index](inputs, positions)
                tolerance = .02 if dtype == torch.bfloat16 else 2e-6
                torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
                comparisons.append({"mlp": "Mlp" if mlp is None else "SwiGLU", "block": index,
                                    "dtype": str(dtype), "max_abs_error": float((actual - expected).abs().max())})
        # The upstream no-position path passes a list of None placeholders.
        plain = SelfAttentionBlock(32, num_heads=4, qk_norm=True, **kwargs).eval().to(device)
        wrapped = _ViewBatchedBlock(copy.deepcopy(plain), 2).eval()
        inputs = torch.randn((5, 17, 32), device=device)
        with torch.no_grad():
            torch.testing.assert_close(wrapped(inputs, [None] * 5), plain(inputs, [None] * 5),
                                       rtol=2e-6, atol=2e-6)
        configure_mapanything_batches(model)
        assert set(model.state_dict()) == keys

    for registers in (False, True):
        model = EncoderModel(registers).eval().to(device)
        views = [{"img": torch.randn((2, 3, 7, 9), device=device)} for _ in range(5)]
        original = model._encode_n_views
        weights = {key: value.cpu().clone() for key, value in model.encoder.state_dict().items()}
        with torch.no_grad():
            expected = original(views)
            configure_mapanything_batches(model, 2, offload_encoder=True)
            actual = model._encode_n_views(views)
        for items, targets in zip(actual, expected):
            if targets is None:
                assert items is None
            else:
                assert len(items) == len(targets) == len(views)
                for value, target in zip(items, targets):
                    assert value.device == target.device
                    torch.testing.assert_close(value, target, rtol=2e-6, atol=2e-6)
        assert all(parameter.device.type == "cpu" for parameter in model.encoder.parameters())
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        configure_mapanything_batches(model)
        assert model._encode_n_views == original
    return {"passed": True, "device": str(device), "comparisons": comparisons,
            "encoder_order_and_registers": True, "weights_preserved": True, "default_graph_restored": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = check_batches(args.device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
