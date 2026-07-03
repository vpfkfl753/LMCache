# SPDX-License-Identifier: Apache-2.0
"""Pure-Python fallback for LMCache native_storage_ops.

The pinned LMCache source includes a Python 3.10 native extension artifact in
this checkout. The project runtime currently uses Python 3.12, so the extension
cannot be imported. This fallback implements the small API surface used by the
MP server smoke path. It prioritizes correctness and importability over native
performance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import os
import struct
import threading
import time


class TTLLock:
    def __init__(self, ttl_second: int = 300) -> None:
        self._ttl_second = max(0, int(ttl_second))
        self._count = 0
        self._expires_at = 0.0
        self._lock = threading.Lock()

    def _expired(self, now: float | None = None) -> bool:
        if self._count <= 0:
            return True
        if self._ttl_second <= 0:
            return False
        return (now if now is not None else time.monotonic()) >= self._expires_at

    def lock(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._expired(now):
                self._count = 0
            self._count += 1
            self._expires_at = now + self._ttl_second

    def unlock(self) -> None:
        with self._lock:
            if self._expired():
                self._count = 0
            elif self._count > 0:
                self._count -= 1
            if self._count == 0:
                self._expires_at = 0.0

    def is_locked(self) -> bool:
        with self._lock:
            if self._expired():
                self._count = 0
                self._expires_at = 0.0
            return self._count > 0

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._expires_at = 0.0


class Bitmap:
    def __init__(self, size: int, prefix_bits: int = 0) -> None:
        self.size = max(0, int(size))
        prefix = min(max(0, int(prefix_bits)), self.size)
        self._bits = set(range(prefix))

    def set(self, index: int) -> None:
        index = int(index)
        if 0 <= index < self.size:
            self._bits.add(index)

    def batched_set(self, indices: Sequence[int]) -> None:
        for index in indices:
            self.set(int(index))

    def set_range(self, start: int, end: int) -> None:
        start = max(0, int(start))
        end = min(self.size, int(end))
        if end > start:
            self._bits.update(range(start, end))

    def clear(self, index: int) -> None:
        self._bits.discard(int(index))

    def test(self, index: int) -> bool:
        index = int(index)
        return 0 <= index < self.size and index in self._bits

    def popcount(self) -> int:
        return len(self._bits)

    def count_leading_zeros(self) -> int:
        count = 0
        for index in range(self.size):
            if self.test(index):
                break
            count += 1
        return count

    def count_leading_ones(self) -> int:
        count = 0
        for index in range(self.size):
            if not self.test(index):
                break
            count += 1
        return count

    def highest_set_bit(self) -> int:
        return max(self._bits) if self._bits else -1

    def __and__(self, other: "Bitmap") -> "Bitmap":
        size = min(self.size, other.size)
        out = Bitmap(size)
        out._bits = {index for index in self._bits & other._bits if index < size}
        return out

    def __or__(self, other: "Bitmap") -> "Bitmap":
        size = min(self.size, other.size)
        out = Bitmap(size)
        out._bits = {index for index in self._bits | other._bits if index < size}
        return out

    def __invert__(self) -> "Bitmap":
        out = Bitmap(self.size)
        out._bits = set(range(self.size)) - self._bits
        return out

    def get_indices_list(self) -> list[int]:
        return sorted(self._bits)

    def get_indices_set(self) -> set[int]:
        return set(self._bits)

    def gather(self, items: Sequence[Any]) -> list[Any]:
        return [items[index] for index in self.get_indices_list() if index < len(items)]

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return "".join(
            "1" if index in self._bits else "0" for index in range(self.size)
        )


def fold(
    found: Bitmap,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1 (got {num_ranks})")
    if not group_windows:
        raise ValueError("group_windows must be non-empty")
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be >= 0 (got {num_chunks})")

    group_stride = num_chunks * num_ranks
    servable = [True] * num_chunks
    for group_idx, window in enumerate(group_windows):
        base = group_idx * group_stride
        effective_window = num_chunks if window <= 0 else int(window)
        run = 0
        for prefix_len in range(1, num_chunks + 1):
            chunk_base = base + (prefix_len - 1) * num_ranks
            present = all(found.test(chunk_base + rank) for rank in range(num_ranks))
            run = run + 1 if present else 0
            if servable[prefix_len - 1] and run < min(effective_window, prefix_len):
                servable[prefix_len - 1] = False

    out = Bitmap(num_chunks)
    for index, is_servable in enumerate(servable):
        if is_servable:
            out.set(index)
    return out


def unfold(
    hit_length: int,
    num_chunks: int,
    num_ranks: int,
    group_windows: Sequence[int],
) -> Bitmap:
    if num_ranks < 1:
        raise ValueError(f"num_ranks must be >= 1 (got {num_ranks})")
    if not group_windows:
        raise ValueError("group_windows must be non-empty")
    if num_chunks < 0:
        raise ValueError(f"num_chunks must be >= 0 (got {num_chunks})")

    hit_length = min(max(0, int(hit_length)), num_chunks)
    group_stride = num_chunks * num_ranks
    out = Bitmap(len(group_windows) * group_stride)
    for group_idx, window in enumerate(group_windows):
        start = 0 if window <= 0 else max(0, hit_length - int(window))
        base = group_idx * group_stride
        for chunk_idx in range(start, hit_length):
            chunk_base = base + chunk_idx * num_ranks
            for rank in range(num_ranks):
                out.set(chunk_base + rank)
    return out


class ParallelPatternMatcher:
    def __init__(self, pattern: list[int]) -> None:
        if not pattern:
            raise ValueError("pattern must not be empty")
        self._pattern = list(pattern)

    def match(self, data: list[int]) -> list[int]:
        n = len(self._pattern)
        return [
            index
            for index in range(0, len(data) - n + 1)
            if data[index : index + n] == self._pattern
        ]


class RangePatternMatcher:
    def __init__(self, start_pattern: list[int], end_pattern: list[int]) -> None:
        if not start_pattern or not end_pattern:
            raise ValueError("patterns must not be empty")
        if len(start_pattern) > 5 or len(end_pattern) > 5:
            raise ValueError("patterns must have at most 5 elements")
        self._start = list(start_pattern)
        self._end = list(end_pattern)

    def match(self, data: list[int]) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start_len = len(self._start)
        end_len = len(self._end)
        index = 0
        while index <= len(data) - start_len:
            if data[index : index + start_len] != self._start:
                index += 1
                continue
            cursor = index + start_len
            while cursor <= len(data) - end_len:
                if data[cursor : cursor + end_len] == self._end:
                    ranges.append((index, cursor + end_len))
                    break
                cursor += 1
            index += 1
        return ranges


class PeriodicEventNotifier:
    _instance: "PeriodicEventNotifier | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, interval_ms: int, use_eventfd: bool) -> None:
        self._interval_s = max(1, int(interval_ms)) / 1000.0
        self._use_eventfd = bool(use_eventfd)
        self._fds: set[int] = set()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def create(interval_ms: int, use_eventfd: bool) -> None:
        with PeriodicEventNotifier._instance_lock:
            if PeriodicEventNotifier._instance is None:
                PeriodicEventNotifier._instance = PeriodicEventNotifier(
                    interval_ms,
                    use_eventfd,
                )

    @staticmethod
    def get() -> "PeriodicEventNotifier | None":
        return PeriodicEventNotifier._instance

    @staticmethod
    def shutdown() -> None:
        with PeriodicEventNotifier._instance_lock:
            inst = PeriodicEventNotifier._instance
            PeriodicEventNotifier._instance = None
        if inst is not None:
            inst._stop.set()
            inst._wake.set()
            inst._thread.join(timeout=2.0)

    def register_fd(self, fd: int) -> None:
        with self._lock:
            self._fds.add(int(fd))
        self._wake.set()

    def unregister_fd(self, fd: int) -> None:
        with self._lock:
            self._fds.discard(int(fd))

    def set_interval_ms(self, interval_ms: int) -> None:
        self._interval_s = max(1, int(interval_ms)) / 1000.0
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self._interval_s)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                fds = list(self._fds)
            if not fds:
                continue
            payload = struct.pack("Q", 1) if self._use_eventfd else b"\x01"
            for fd in fds:
                try:
                    os.write(fd, payload)
                except OSError:
                    self.unregister_fd(fd)
