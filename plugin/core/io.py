"""JSONL 읽기/쓰기와 Synapse Storage 연동.

결과 파일을 임시 폴더에 쓴 뒤 SDK의 Storage로 올린다. storage 설정만 바꾸면
로컬 폴더 대신 S3·MinIO·GCS·SFTP 같은 저장소에 그대로 저장된다.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from synapse_sdk.utils.storage import get_pathlib, get_storage


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            data = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_jsonl(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in lines:
        if line.strip():
            yield json.loads(line)


def upload_files(storage_config: dict[str, Any], prefix: str, files: dict[str, Path]) -> dict[str, str]:
    """{파일명: 로컬 경로}를 storage의 prefix 아래로 올리고 {파일명: 저장 위치}를 돌려준다."""
    storage = get_storage(storage_config)
    return {name: storage.upload(path, f"{prefix.strip('/')}/{name}") for name, path in files.items()}


def read_text_lines(storage_config: dict[str, Any], key: str) -> Iterator[str]:
    path = get_pathlib(storage_config, key)
    with path.open("r", encoding="utf-8") as f:
        yield from f


class TempDir:
    """with 블록이 끝나면 지워지는 작업 폴더."""

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory(prefix="paper-rag-")
        return Path(self._tmp.name)

    def __exit__(self, *exc: object) -> None:
        self._tmp.cleanup()
