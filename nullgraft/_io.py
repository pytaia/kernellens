"""Ввод-вывод: атомарная запись и отпечатки сборок для возобновляемых чекпоинтов."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def atomic_write_text(path: str | Path, text: str) -> None:
    """Запись через временный файл и ``os.replace``: сбой не оставит недописанный файл."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_npy(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, np.asarray(arr))
    os.replace(tmp, path)


def fingerprint(params: dict[str, Any]) -> str:
    """Стабильный хэш параметров сборки (не-JSON значения — через ``repr``)."""
    return hashlib.sha1(json.dumps(params, sort_keys=True, default=repr).encode()).hexdigest()


def slices_digest(slices: Any, max_probe: int = 3) -> str:
    """Дайджест набора срезов: число, формы и содержимое нескольких срезов (первый, последний, средние)."""
    n = len(slices)
    h = hashlib.sha1(f"{n}".encode())
    if n:
        for i in sorted({0, n - 1, *np.linspace(0, n - 1, min(n, max_probe)).astype(int).tolist()}):
            arr = np.ascontiguousarray(np.asarray(slices[i], np.float32))
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()
