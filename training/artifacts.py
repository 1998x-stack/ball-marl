"""Crash-safe experiment artifacts and SHA256 corruption checks.

SHA256 is not a cryptographic publisher signature.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def atomic_write_bytes(path, content):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{target.name}.', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
    finally:
        if os.path.exists(name):
            os.unlink(name)

def atomic_write_json(path, data):
    content = json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2,
                         allow_nan=False).encode('utf-8') + b'\n'
    atomic_write_bytes(path, content)

def verify_checksum(path, *, allow_legacy=False):
    target = Path(path)
    sidecar = Path(str(target) + '.sha256')
    if not sidecar.is_file():
        if allow_legacy:
            return False
        raise ValueError(f'checkpoint lacks checksum: {sidecar}; use allow_legacy=True explicitly')
    expected = sidecar.read_text(encoding='ascii').strip()
    if len(expected) != 64 or not all(c in '0123456789abcdef' for c in expected):
        raise ValueError('checkpoint checksum sidecar malformed')
    if sha256_file(target) != expected:
        raise ValueError('checkpoint checksum mismatch: file corrupted or replaced')
    return True
