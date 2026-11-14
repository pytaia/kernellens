"""Фоновые задачи с прогрессом и отменой, кэш результатов в памяти и на диске. Без GUI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

ProgressFn = Callable[[float, str], None]


class Cancelled(Exception):
    """Задача отменена пользователем."""


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled()


@dataclass
class Job:
    """Запущенная задача: ``cancel()`` выставляет токен и, если задача ещё не началась, снимает её."""

    name: str
    future: Future
    token: CancelToken
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def cancel(self) -> None:
        self.token.cancel()
        self.future.cancel()

    def done(self) -> bool:
        return self.future.done()

    def result(self, timeout: float | None = None) -> Any:
        return self.future.result(timeout)


class JobRunner:
    """Пул потоков для задач с прогрессом; пул процессов — для тяжёлых чистых функций без прогресса."""

    def __init__(self, max_workers: int = 2, process_workers: int = 0) -> None:
        self._threads = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nullgraft-job")
        self._process_workers = process_workers
        self._processes: ProcessPoolExecutor | None = None

    def submit(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
        on_progress: ProgressFn | None = None,
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Job:
        """``fn(*args, token=..., progress=..., **kwargs)`` в фоне; колбэки вызываются из рабочего потока."""
        token = CancelToken()

        def progress(fraction: float, message: str = "") -> None:
            token.check()
            if on_progress is not None:
                on_progress(min(max(fraction, 0.0), 1.0), message)

        def run() -> Any:
            try:
                result = fn(*args, token=token, progress=progress, **kwargs)
                token.check()
            except Cancelled:
                logger.info("job %s cancelled", name)
                if on_cancel is not None:
                    on_cancel()
                raise
            except BaseException as exc:
                logger.exception("job %s failed", name)
                if on_error is not None:
                    on_error(exc)
                raise
            if on_done is not None:
                on_done(result)
            return result

        return Job(name, self._threads.submit(run), token)

    def submit_process(self, name: str, fn: Callable[..., Any], *args: Any,
                       on_done: Callable[[Any], None] | None = None,
                       on_error: Callable[[BaseException], None] | None = None) -> Job:
        """Чистая функция в отдельном процессе (аргументы и результат должны сериализоваться)."""
        if self._processes is None:
            workers = self._process_workers or max(1, (os.cpu_count() or 2) - 1)
            self._processes = ProcessPoolExecutor(max_workers=workers)
        token = CancelToken()
        fut = self._processes.submit(fn, *args)

        def finished(f: Future) -> None:
            if token.cancelled or f.cancelled():
                return
            exc = f.exception()
            if exc is not None:
                if on_error is not None:
                    on_error(exc)
            elif on_done is not None:
                on_done(f.result())

        fut.add_done_callback(finished)
        return Job(name, fut, token)

    def shutdown(self, wait: bool = False) -> None:
        self._threads.shutdown(wait=wait, cancel_futures=True)
        if self._processes is not None:
            self._processes.shutdown(wait=wait, cancel_futures=True)


def cache_key(*parts: Any) -> str:
    """SHA1 по частям: массивы — по форме, типу и байтам, остальное — через JSON."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            arr = np.ascontiguousarray(p)
            h.update(f"{arr.shape}{arr.dtype}".encode())
            h.update(arr.tobytes())
        else:
            h.update(json.dumps(p, sort_keys=True, default=repr).encode())
    return h.hexdigest()


class ResultCache:
    """Кэш наборов массивов по ключу: LRU в памяти и ``.npz`` на диске (если задан каталог)."""

    def __init__(self, directory: str | Path | None = None, memory_items: int = 32) -> None:
        self.directory = Path(directory) if directory else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.memory_items = memory_items
        self._mem: OrderedDict[str, dict[str, NDArray]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path | None:
        return None if self.directory is None else self.directory / f"{key}.npz"

    def get(self, key: str) -> dict[str, NDArray] | None:
        with self._lock:
            if key in self._mem:
                self._mem.move_to_end(key)
                self.hits += 1
                return dict(self._mem[key])
        path = self._path(key)
        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in data.files}
            except (OSError, ValueError) as exc:
                logger.warning("broken cache entry %s: %s", path.name, exc)
                return None
            self._remember(key, arrays)
            with self._lock:
                self.hits += 1
            return dict(arrays)
        with self._lock:
            self.misses += 1
        return None

    def put(self, key: str, arrays: dict[str, NDArray]) -> None:
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self._remember(key, arrays)
        path = self._path(key)
        if path is not None:
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            os.replace(tmp, path)

    def _remember(self, key: str, arrays: dict[str, NDArray]) -> None:
        with self._lock:
            self._mem[key] = arrays
            self._mem.move_to_end(key)
            while len(self._mem) > self.memory_items:
                self._mem.popitem(last=False)
