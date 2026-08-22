import os
import shutil
from pathlib import Path


def _is_rank_zero() -> bool:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value not in (None, "", "0"):
            return False
    return True


def prepare_huggingface_checkpoint(
    repo_id: str,
    checkpoint_filename: str,
    local_checkpoint_path: str,
    counter_filename: str = "config.json",
    revision: str = "main",
    rank_zero_only: bool = True,
) -> str:
    """Ensure the local MONAI checkpoint path exists and count a first-time download."""

    local_path = Path(local_checkpoint_path)

    if rank_zero_only and not _is_rank_zero():
        return str(local_path)

    if local_path.exists():
        return str(local_path)

    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=counter_filename,
        repo_type="model",
        revision=revision,
        force_download=True,
    )
    print(f"[nvseg] registered Hugging Face download for {repo_id}/{counter_filename}")

    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=checkpoint_filename,
        repo_type="model",
        revision=revision,
    )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        local_path.symlink_to(checkpoint_path)
    except OSError:
        shutil.copy2(checkpoint_path, local_path)

    print(f"[nvseg] prepared checkpoint at {local_path}")
    return str(local_path)
