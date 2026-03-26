"""
Cohort batch inference: discover NIfTI inputs, match MONAI SaveImaged paths, optional resume.

Behavior is controlled from ``configs/batch_inference.json`` (skip dir lists, resume, cache).
To change which classes are segmented, edit ``everything_labels`` in ``configs/inference.json``.
To customize path filtering further, edit :func:`should_skip_path_by_parent_rules`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def should_skip_path_by_parent_rules(
    path: Path,
    *,
    skip_dir_names: list[str] | None = None,
    skip_dir_prefixes: list[str] | None = None,
) -> bool:
    """
    Return True if this path should be excluded from batch discovery.

    Default (empty lists): do not skip any path based on directory names.

    - ``skip_dir_names``: any **parent** directory component that equals a name (case-insensitive).
    - ``skip_dir_prefixes``: any **parent** directory component whose name **starts with** a prefix
      (case-insensitive).
    """
    names = {n.strip().lower() for n in (skip_dir_names or []) if n and str(n).strip()}
    prefixes = tuple(p.strip().lower() for p in (skip_dir_prefixes or []) if p and str(p).strip())
    for part in path.parts[:-1]:
        pl = part.lower()
        if pl in names:
            return True
        for prefix in prefixes:
            if prefix and pl.startswith(prefix):
                return True
    return False


def expected_output_path(
    input_path: Path,
    input_root: Path,
    output_dir: Path,
    postfix: str,
    ext: str,
) -> Path:
    """Match MONAI FolderLayout + separate_folder + data_root_dir."""
    input_path = input_path.resolve()
    input_root = input_root.resolve()
    output_dir = output_dir.resolve()
    rel = os.path.relpath(input_path, input_root)
    rel_dir = os.path.dirname(rel)
    stem = input_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[: -len(".nii.gz")]
    elif stem.endswith(".nii"):
        stem = stem[: -len(".nii")]
    sub = Path(rel_dir) if rel_dir else Path()
    return output_dir / sub / stem / f"{stem}_{postfix}{ext}"


def collect_input_paths(
    input_root: Path,
    pattern: str = "**/*.nii.gz",
    *,
    skip_dir_names: list[str] | None = None,
    skip_dir_prefixes: list[str] | None = None,
) -> list[Path]:
    input_root = input_root.resolve()
    out: list[Path] = []
    for p in sorted(input_root.glob(pattern)):
        if not p.is_file():
            continue
        if should_skip_path_by_parent_rules(
            p, skip_dir_names=skip_dir_names, skip_dir_prefixes=skip_dir_prefixes
        ):
            continue
        out.append(p)
    return out


def _cache_path(
    input_dir: str,
    output_dir: str,
    postfix: str,
    ext: str,
    skip: bool,
    skip_dir_names: list[str] | None,
    skip_dir_prefixes: list[str] | None,
) -> Path:
    sig = (
        f"{os.path.abspath(input_dir)}|{os.path.abspath(output_dir)}|{postfix}|{ext}|{skip}|"
        f"{sorted(skip_dir_names or [])}|{sorted(skip_dir_prefixes or [])}"
    )
    h = hashlib.sha256(sig.encode()).hexdigest()[:24]
    base = Path(tempfile.gettempdir())
    return base / f"nvseg_batch_input_{h}.json"


def _compute_input_list(
    input_dir: str,
    output_dir: str,
    postfix: str,
    ext: str,
    *,
    skip_existing: bool,
    skip_dir_names: list[str] | None,
    skip_dir_prefixes: list[str] | None,
) -> tuple[list[str], int]:
    """
    Returns ``(paths_to_run, n_discovered)`` where ``n_discovered`` is the number of
    ``*.nii.gz`` paths after directory filters (before resume skip).
    """
    root = Path(input_dir)
    out_root = Path(output_dir)
    all_paths = collect_input_paths(
        root,
        skip_dir_names=skip_dir_names,
        skip_dir_prefixes=skip_dir_prefixes,
    )
    n_discovered = len(all_paths)
    if not skip_existing:
        return [str(p) for p in all_paths], n_discovered

    missing: list[str] = []
    for inp in all_paths:
        exp = expected_output_path(inp, root, out_root, postfix, ext)
        if not exp.is_file() or exp.stat().st_size == 0:
            missing.append(str(inp))
    return missing, n_discovered


def _parse_cache_payload(raw: Any) -> tuple[list[str], int]:
    """Load cache written by rank 0. Supports legacy JSON list for backward compatibility."""
    if isinstance(raw, list):
        # Legacy: empty [] is ambiguous (stale file or old format) -> signal recompute on workers.
        return [str(p) for p in raw], len(raw) if raw else -1
    if isinstance(raw, dict) and "paths" in raw:
        paths = raw["paths"]
        if not isinstance(paths, list):
            raise RuntimeError("[nvseg] batch: bad cache (paths); rm /tmp/nvseg_batch_input_*.json")
        n_raw = raw.get("n_discovered")
        if n_raw is None:
            n_discovered = len(paths) if paths else -1
        else:
            n_discovered = int(n_raw)
        return [str(p) for p in paths], n_discovered
    raise RuntimeError("[nvseg] batch: bad cache format; rm /tmp/nvseg_batch_input_*.json")


def build_input_list(
    input_dir: str,
    output_dir: str,
    output_postfix: str,
    output_ext: str,
    batch_skip_dir_names: list | None = None,
    batch_skip_dir_prefixes: list | None = None,
    batch_resume_skip_existing: bool = True,
    batch_use_input_list_cache: bool = True,
    batch_cache_wait_sec: float = 120.0,
) -> list[str]:
    """
    Called from ``configs/batch_inference.json``.

    ``batch_resume_skip_existing``: if True, only queue inputs whose output file is missing or empty.
    ``batch_skip_dir_names`` / ``batch_skip_dir_prefixes``: filter discovery (see
    :func:`should_skip_path_by_parent_rules`).

    ``LOCAL_RANK`` (set by ``torchrun``) is still read from the environment for multi-GPU cache.

    If resume mode leaves nothing to run (outputs already exist), prints a message and raises
    ``SystemExit(0)`` so the process exits before MONAI builds a zero-length DataLoader (which
    would fail under ``DistributedSampler``). If no ``*.nii.gz`` files are discovered, raises
    ``RuntimeError``.
    """
    names = list(batch_skip_dir_names) if batch_skip_dir_names is not None else []
    prefixes = list(batch_skip_dir_prefixes) if batch_skip_dir_prefixes is not None else []

    skip = bool(batch_resume_skip_existing)
    use_cache = bool(batch_use_input_list_cache)
    wait_sec = float(batch_cache_wait_sec)

    local_rank = os.environ.get("LOCAL_RANK", "0")

    if not use_cache or local_rank == "0":
        paths, n_discovered = _compute_input_list(
            input_dir,
            output_dir,
            output_postfix,
            output_ext,
            skip_existing=skip,
            skip_dir_names=names,
            skip_dir_prefixes=prefixes,
        )
        payload = {"paths": paths, "n_discovered": n_discovered}
        if use_cache and local_rank == "0":
            cache = _cache_path(
                input_dir, output_dir, output_postfix, output_ext, skip, names, prefixes
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(cache)

        if local_rank == "0":
            mode = "resume (skip existing outputs)" if skip else "full pass (all inputs)"
            print(
                f"[nvseg] batch {mode}: {len(paths)} volume(s) "
                f"(input_dir={os.path.abspath(input_dir)}, output_dir={os.path.abspath(output_dir)})",
                flush=True,
            )
    else:
        cache = _cache_path(
            input_dir, output_dir, output_postfix, output_ext, skip, names, prefixes
        )
        deadline = time.time() + wait_sec
        payload = None
        while time.time() < deadline:
            if cache.is_file():
                payload = json.loads(cache.read_text())
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(
                "[nvseg] batch: cache timeout (raise batch_cache_wait_sec or set batch_use_input_list_cache false)"
            )
        paths, n_discovered = _parse_cache_payload(payload)
        # Stale legacy `[]` or missing n_discovered: recompute so workers agree with rank 0.
        if n_discovered < 0:
            paths, n_discovered = _compute_input_list(
                input_dir,
                output_dir,
                output_postfix,
                output_ext,
                skip_existing=skip,
                skip_dir_names=names,
                skip_dir_prefixes=prefixes,
            )

    if not paths:
        if n_discovered == 0:
            raise RuntimeError("[nvseg] batch: no *.nii.gz under input_dir (check paths and skip rules)")
        # Resume: all outputs exist — exit before an empty DistributedSampler / dataloader.
        print("[nvseg] batch: nothing to run (resume); ok", flush=True)
        raise SystemExit(0)

    return paths


__all__ = [
    "build_input_list",
    "collect_input_paths",
    "expected_output_path",
    "should_skip_path_by_parent_rules",
]
