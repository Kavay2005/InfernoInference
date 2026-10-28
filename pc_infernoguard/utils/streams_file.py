"""Read/write Ultralytics-compatible .streams files; safe IDs and log redaction."""

import re
from pathlib import Path
from typing import Collection, Dict, List, Optional, Tuple, Union
from urllib.parse import urlsplit

import numpy as np

from pc_infernoguard.capture.stream_types import FrameData


def _validate_lines(text: str, origin: str) -> List[str]:
    """Enforce what Ultralytics LoadStreams assumes: whitespace-split tokens, no comments."""
    sources: List[str] = []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            raise ValueError(
                f"{origin}:{n}: comments are not supported; Ultralytics would treat "
                "each word as a stream URL."
            )
        if len(s.split()) != 1:
            raise ValueError(
                f"{origin}:{n}: expected one URI per line with no whitespace "
                "(Ultralytics splits on whitespace)."
            )
        sources.append(s)
    if not sources:
        raise ValueError(f"{origin}: no stream sources found")
    dupes = {s for s in sources if sources.count(s) > 1}
    if dupes:
        raise ValueError(f"{origin}: duplicate sources would open duplicate connections")
    return sources


def parse_streams_file(file_path: Union[str, Path]) -> List[str]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Streams file not found: {file_path}")
    return _validate_lines(path.read_text(encoding="utf-8-sig"), str(path))  # strips BOM


def generate_streams_file(sources: List[Union[str, int]], file_path: Union[str, Path]) -> Path:
    path = Path(file_path)
    text = "\n".join(str(s).strip() for s in sources if str(s).strip())
    lines = _validate_lines(text, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")  # no BOM
    try:
        path.chmod(0o600)  # file may contain camera credentials
    except OSError:
        pass
    return path


def redact_uri(uri: Union[str, int]) -> str:
    """Mask user:password@ and query strings so URIs are safe to log."""
    s = str(uri)
    try:
        p = urlsplit(s)
        if p.username is None and p.password is None and not p.query:
            return s
        host = p.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        try:
            port = f":{p.port}" if p.port else ""
        except ValueError:
            port = ""
        netloc = (f"***@{host}" if (p.username or p.password) else host) + port
        return p._replace(netloc=netloc, query="***" if p.query else "").geturl()
    except ValueError:
        return "<unparseable-uri>"


def source_id_for(
    uri: Union[str, int], index: int, taken: Optional[Collection[str]] = None
) -> str:
    """Stable, credential-free ID from host/port/path (independent of file order)."""
    s = str(uri).strip()
    if s.isdigit():
        base = f"webcam_{s}"
    else:
        try:
            p = urlsplit(s)
            host = p.hostname
            port = p.port
        except ValueError:
            host, port, p = None, None, None
        if host:
            base = host + (f"_{port}" if port else "")
            path = p.path.strip("/")
            if path:
                base += "_" + path
        else:
            base = Path(s).stem
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("_")[:48] or f"source_{index}"
    sid, n = base, 2
    while taken and sid in taken:
        sid, n = f"{base}_{n}", n + 1
    return sid


def to_ultralytics_batch(batch: Dict[str, FrameData]) -> Tuple[List[str], List[np.ndarray]]:
    """Unchanged; deprecated once the demo uses predict_batch(frames)."""
    ids = list(batch.keys())
    return ids, [batch[i].image for i in ids]
