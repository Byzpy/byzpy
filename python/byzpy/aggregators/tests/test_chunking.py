import pytest

from byzpy.aggregators._chunking import select_adaptive_chunk_size


def test_select_adaptive_chunk_size_no_pool():
    assert select_adaptive_chunk_size(total_items=1000, configured_chunk=256, pool_size=None) == 256


def test_select_adaptive_chunk_size_caps_to_total():
    assert select_adaptive_chunk_size(total_items=10, configured_chunk=256, pool_size=None) == 10


def test_select_adaptive_chunk_size_shrinks_with_pool():
    chunk = select_adaptive_chunk_size(
        total_items=131072,
        configured_chunk=8192,
        pool_size=8,
        min_chunks_per_worker=4,
    )
    # plenty of workers -> chunk size drops to keep schedulable subtasks
    assert chunk < 8192
    assert chunk >= 8192 // 8


def test_select_adaptive_chunk_size_env_min_chunks(monkeypatch):
    monkeypatch.setenv("BYZPY_CHUNK_MIN_PER_WORKER", "16")
    chunk = select_adaptive_chunk_size(total_items=1048576, configured_chunk=65536, pool_size=4)
    assert chunk < 65536
    monkeypatch.delenv("BYZPY_CHUNK_MIN_PER_WORKER", raising=False)


def test_select_adaptive_chunk_size_env_target_factor(monkeypatch):
    monkeypatch.setenv("BYZPY_CHUNK_TARGET_FACTOR", "3.0")
    chunk = select_adaptive_chunk_size(total_items=131072, configured_chunk=32768, pool_size=2)
    assert chunk < 32768
    monkeypatch.delenv("BYZPY_CHUNK_TARGET_FACTOR", raising=False)
