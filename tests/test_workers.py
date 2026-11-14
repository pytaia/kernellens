from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from nullgraft_demo.compute import reconstruct, suggest_candidates
from nullgraft_demo.datasource import open_source, synthetic_source
from nullgraft_demo.state import ReconParams
from nullgraft_demo.workers import Cancelled, CancelToken, JobRunner, ResultCache, cache_key


def _slow(n: int, *, token: CancelToken, progress) -> int:
    for i in range(n):
        progress(i / n, f"шаг {i}")
        time.sleep(0.01)
    return n


@pytest.fixture()
def runner():
    r = JobRunner(max_workers=2)
    yield r
    r.shutdown(wait=True)


def test_job_reports_progress_and_result(runner: JobRunner) -> None:
    seen: list[float] = []
    done = threading.Event()
    out: list[int] = []
    job = runner.submit("slow", _slow, 5, on_progress=lambda f, m: seen.append(f),
                        on_done=lambda r: (out.append(r), done.set()))
    assert job.result(timeout=5) == 5 and done.wait(5)
    assert out == [5] and seen == sorted(seen) and len(seen) == 5


def test_job_can_be_cancelled(runner: JobRunner) -> None:
    cancelled = threading.Event()
    job = runner.submit("slow", _slow, 500, on_cancel=cancelled.set)
    time.sleep(0.05)
    job.cancel()
    with pytest.raises(Cancelled):
        job.result(timeout=5)
    assert cancelled.wait(5) and job.token.cancelled


def test_job_error_is_reported(runner: JobRunner) -> None:
    errors: list[BaseException] = []

    def boom(*, token, progress):
        raise RuntimeError("boom")

    job = runner.submit("boom", boom, on_error=errors.append)
    with pytest.raises(RuntimeError):
        job.result(timeout=5)
    assert isinstance(errors[0], RuntimeError)


def _square(v: int) -> int:
    return v * v


def test_process_job(runner: JobRunner) -> None:
    got: list[int] = []
    ev = threading.Event()
    job = runner.submit_process("sq", _square, 7, on_done=lambda r: (got.append(r), ev.set()))
    assert job.result(timeout=60) == 49 and ev.wait(5) and got == [49]


def test_cache_key_is_content_based() -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert cache_key("x", a, 1) == cache_key("x", a.copy(), 1)
    assert cache_key("x", a, 1) != cache_key("x", a.T.copy(), 1)
    assert cache_key("x", a, 1) != cache_key("x", a, 2)


def test_result_cache_memory_and_disk(tmp_path) -> None:
    cache = ResultCache(tmp_path, memory_items=1)
    cache.put("k1", {"a": np.ones(3)})
    cache.put("k2", {"a": np.zeros(3)})
    assert (tmp_path / "k1.npz").exists()
    np.testing.assert_array_equal(cache.get("k1")["a"], np.ones(3))  # вытеснен из памяти, поднят с диска
    fresh = ResultCache(tmp_path)
    np.testing.assert_array_equal(fresh.get("k2")["a"], np.zeros(3))
    assert fresh.get("missing") is None and fresh.misses == 1
    (tmp_path / "bad.npz").write_bytes(b"not a zip")
    assert fresh.get("bad") is None


def test_reconstruction_is_cached(tmp_path) -> None:
    src = synthetic_source(size=48, n_slices=2)
    params = ReconParams(n_angles=20, decoder="fbp-hann")
    cache = ResultCache(tmp_path)
    token = CancelToken()
    msgs: list[str] = []
    first = reconstruct(src.get(1), params, cache, token=token, progress=lambda f, m: msgs.append(m))
    second = reconstruct(src.get(1), params, cache, token=token, progress=lambda f, m: None)
    assert not first.from_cache and second.from_cache and second.key == first.key
    np.testing.assert_array_equal(first.z, second.z)
    third = reconstruct(src.get(1), params.with_(seed=1), cache, token=token, progress=lambda f, m: None)
    assert not third.from_cache
    assert first.det_residual.shape == first.y.shape and first.eps > 0 and msgs[-1]


def test_cancelled_token_stops_reconstruction() -> None:
    src = synthetic_source(size=32, n_slices=1)
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        reconstruct(src.get(0), ReconParams(n_angles=10, decoder="fbp-ramp"), None, token=token,
                    progress=lambda f, m: token.check())


def test_candidates_find_the_strongest_residual() -> None:
    x = np.zeros((40, 40), np.float32)
    z = x.copy()
    z[25:29, 10:14] = 1.0
    cands = suggest_candidates(z, x, k=3, radius=4)
    r, c, rad, score = cands[0]
    assert abs(r - 26.5) <= 1.5 and abs(c - 11.5) <= 1.5 and rad == 4 and score > 0
    assert suggest_candidates(z, None, k=2, radius=4)


def test_open_source_variants(tmp_path) -> None:
    cfg = {"synthetic": {"size": 32, "n_slices": 3}, "pixel_mm": 0.7}
    syn = open_source(None, cfg)
    assert syn.n_slices == 3 and syn.has_truth and syn.pixel_mm == 0.7
    np.save(tmp_path / "one.npy", np.ones((10, 14), np.float32))
    one = open_source(tmp_path / "one.npy", cfg)
    assert one.get(0).shape == (14, 14) and not one.has_truth
    with pytest.raises(FileNotFoundError):
        open_source(tmp_path, cfg)
    with pytest.raises(IndexError):
        one.get(5)
