"""Read files from a Hugging Face model repo or its downloaded directory."""

import json
from pathlib import Path


def resolve_model_sources(plsr_repo_id=None, trellis2_repo_id=None, *, dino_repo_id=None,
                          mapanything_repo_id=None, rmbg_repo_id=None):
    """Explicit API arguments override the entry's environment-backed defaults."""
    from . import __main__ as entry

    return {
        "plsr": str(entry.PLSR_REPO_ID if plsr_repo_id is None else plsr_repo_id),
        "trellis2": str(entry.TRELLIS2_REPO_ID if trellis2_repo_id is None else trellis2_repo_id),
        "dino": str(entry.DINO_REPO_ID if dino_repo_id is None else dino_repo_id),
        "mapanything": str(entry.MAPANYTHING_REPO_ID if mapanything_repo_id is None else mapanything_repo_id),
        "rmbg": str(entry.RMBG_REPO_ID if rmbg_repo_id is None else rmbg_repo_id),
    }


def repo_file(source, filename):
    """Keep the same relative layout for online and downloaded repositories."""
    directory = Path(source).expanduser()
    if directory.is_dir():
        return directory / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=str(source), filename=filename))


def load_inference_defaults(source):
    """Read inference parameters without importing models or loading weights."""
    return json.loads(repo_file(source, "pipeline.json").read_text(encoding="utf-8"))["inference_defaults"]
