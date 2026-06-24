from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

# Default HuggingFace dataset repo (mirrors sora-accidents-dataset.md layout:
# metadata.csv at the root, plus videos/<label>/ and records/<label>/ trees).
DEFAULT_DATASET = "gnitoahc/sora-accidents-copy"


@dataclass(frozen=True)
class MetadataRow:
    """A row from ``metadata.csv`` with a usable generation record."""

    file_name: str  # video ref, repo- or dir-relative
    label: str
    record_file: str  # record JSON ref, repo- or dir-relative


@dataclass(frozen=True)
class DatasetItem:
    """One scored unit: a materialized video paired with its ground-truth prompt.

    Attributes:
        video_path: Local path to the ``.mp4`` to run the model on.
        label: Action label from ``metadata.csv``.
        ground_truth_prompt: The Sora generation prompt that created the
            video, read from the record JSON's ``"prompt"`` field.
        record_file: Local path to the record JSON the prompt came from.
    """

    video_path: Path
    label: str
    ground_truth_prompt: str
    record_file: Path


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t"}


def read_ground_truth_prompt(record_file: Path) -> str:
    """Read the generation prompt from a record JSON file.

    Args:
        record_file: Path to a record JSON containing a ``"prompt"`` field.

    Returns:
        The prompt string.

    Raises:
        KeyError: If the JSON has no non-empty ``"prompt"`` field.
    """
    data = json.loads(record_file.read_text(encoding="utf-8"))
    prompt = data.get("prompt")
    if not prompt:
        raise KeyError(f"No 'prompt' field in record: {record_file}")
    return prompt


class DatasetSource:
    """Resolve dataset members from a local dir or a HuggingFace dataset repo.

    If ``source`` is an existing local path it is used directly; otherwise it
    is treated as a HuggingFace ``dataset`` repo id and members are downloaded
    on demand (and cached by ``huggingface_hub``). Downloading per resolved row
    rather than up front means a small ``sample_size`` never pulls the whole
    ~1.5k-video dataset.
    """

    def __init__(self, source: str | Path, hf_token: str | None = None):
        self.hf_token = hf_token
        local = Path(source)
        self.local_dir: Path | None = local if local.exists() else None
        self.repo_id: str | None = None if self.local_dir else str(source)

    def _fetch(self, ref: str) -> Path:
        """Return a local path for a repo-relative ref, downloading if needed."""
        if self.local_dir is not None:
            path = Path(ref)
            return path if path.is_absolute() else self.local_dir / ref
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(self.repo_id, ref, repo_type="dataset", token=self.hf_token)
        )

    def metadata_rows(self) -> list[MetadataRow]:
        """Read ``metadata.csv`` and return rows that have a generation record.

        Rows with ``has_record`` false are dropped, since they have no
        ground-truth prompt to score against.

        Returns:
            One :class:`MetadataRow` per scorable row.
        """
        metadata_path = self._fetch("metadata.csv")
        rows: list[MetadataRow] = []
        with metadata_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not _is_truthy(row.get("has_record", "")):
                    continue
                rows.append(
                    MetadataRow(
                        file_name=row["file_name"],
                        label=row.get("label", "unknown"),
                        record_file=row["record_file"],
                    )
                )
        return rows

    def resolve(self, row: MetadataRow) -> DatasetItem:
        """Materialize a row: fetch its record and video, read the prompt.

        Args:
            row: A metadata row to materialize.

        Returns:
            A :class:`DatasetItem` with local paths and the ground-truth prompt.

        Raises:
            KeyError, OSError, json.JSONDecodeError: If the record is missing
                or has no ``"prompt"`` field.
        """
        record_file = self._fetch(row.record_file)
        prompt = read_ground_truth_prompt(record_file)
        return DatasetItem(
            video_path=self._fetch(row.file_name),
            label=row.label,
            ground_truth_prompt=prompt,
            record_file=record_file,
        )
