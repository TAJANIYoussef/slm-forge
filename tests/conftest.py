"""Shared fixtures. The whole suite runs on CPU and offline; HF-dependent tests are marked ``hf``."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(var, "1")


@pytest.fixture
def repo_root() -> Path:
    """Repository root (configs/, data/ live here)."""
    return REPO


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Block non-loopback IPv4/IPv6 connections (AF_UNIX stays usable for multiprocessing)."""
    real_connect = socket.socket.connect

    def guarded(self: socket.socket, address: object) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            host = address[0] if isinstance(address, tuple) else address
            if host not in ("127.0.0.1", "::1", "localhost"):
                raise OSError(f"network access blocked in tests: {address!r}")
        return real_connect(self, address)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guarded)
