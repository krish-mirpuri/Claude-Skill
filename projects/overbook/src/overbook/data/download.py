"""Fetch the raw extract and verify it against a recorded checksum."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

from overbook.config import Config, load_config

log = logging.getLogger(__name__)
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: Config | None = None, *, force: bool = False) -> Path:
    """Download the extract to ``cfg.data.raw_path`` and verify its checksum.

    A checksum mismatch is a warning, not an error: the upstream mirror can
    legitimately be updated. It is surfaced loudly so that a change in results
    is never mistaken for a change in code.
    """
    cfg = cfg or load_config()
    dest = cfg.resolve(cfg.data.raw_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        log.info("raw extract already present at %s", dest)
    else:
        log.info("downloading %s", cfg.data.source_url)
        with requests.get(cfg.data.source_url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(_CHUNK):
                    fh.write(chunk)
            tmp.replace(dest)

    digest = sha256_of(dest)
    if digest != cfg.data.sha256:
        log.warning(
            "checksum mismatch for %s\n  expected %s\n  got      %s\n"
            "The upstream file may have changed; results will not match the README.",
            dest,
            cfg.data.sha256,
            digest,
        )
    else:
        log.info("checksum verified (%s)", digest[:12])
    return dest


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(download())
