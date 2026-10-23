"""Resolve the local job CSV or download it from its public Kaggle dataset."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


DATASET_HANDLE = "ravindrasinghrana/job-description-dataset"
DATASET_FILENAME = "job_descriptions.csv"


def resolve_dataset_path(project_root: str | Path) -> Path:
    """Return a configured/local dataset, downloading it from Kaggle if needed."""
    project_root = Path(project_root).resolve()

    configured_path = os.getenv("JOB_DATA_PATH")
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"JOB_DATA_PATH does not exist: {path}")
        return path

    local_path = project_root / DATASET_FILENAME
    if local_path.is_file():
        return local_path

    cache_dir = Path(
        os.getenv("JOB_DATA_DIR", str(project_root / ".dataset_cache"))
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / DATASET_FILENAME
    if cached_path.is_file() and cached_path.stat().st_size > 0:
        return cached_path

    if not os.getenv("KAGGLE_API_TOKEN"):
        try:
            import streamlit as st

            token = st.secrets.get("KAGGLE_API_TOKEN")
        except (ImportError, FileNotFoundError):
            token = None
        if token:
            os.environ["KAGGLE_API_TOKEN"] = str(token)

    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError(
            "The dataset is missing and kagglehub is not installed. "
            "Install dependencies from requirements.txt."
        ) from error

    downloaded_path = Path(
        kagglehub.dataset_download(
            DATASET_HANDLE,
            path=DATASET_FILENAME,
            output_dir=str(cache_dir),
        )
    )
    if downloaded_path.is_file():
        source_path = downloaded_path
    elif downloaded_path.is_dir() and (downloaded_path / DATASET_FILENAME).is_file():
        source_path = downloaded_path / DATASET_FILENAME
    else:
        matches = list(cache_dir.rglob(DATASET_FILENAME))
        if not matches:
            raise FileNotFoundError(
                f"Kaggle download completed but {DATASET_FILENAME} was not found "
                f"under {cache_dir}. Check the Kaggle dataset file name."
            )
        source_path = matches[0]

    if source_path.resolve() != cached_path.resolve():
        shutil.copy2(source_path, cached_path)
    if cached_path.stat().st_size == 0:
        raise OSError(f"Downloaded dataset is empty: {cached_path}")
    return cached_path
