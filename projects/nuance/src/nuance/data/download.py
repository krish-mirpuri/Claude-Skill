"""Fetch every corpus this project uses, and verify it hasn't moved."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

from nuance.config import Config, load_config

log = logging.getLogger(__name__)
_CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dest: Path, expected: str | None, force: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        log.debug("already present: %s", dest.name)
    else:
        log.info("downloading %s", url)
        with requests.get(url, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(_CHUNK):
                    fh.write(chunk)
            tmp.replace(dest)
    if expected:
        digest = sha256_of(dest)
        if digest != expected:
            # Not fatal: upstream mirrors do get updated. But a silent change
            # here would show up as an unexplained change in every metric.
            log.warning(
                "checksum mismatch for %s\n  expected %s\n  got      %s\n"
                "Upstream may have changed; reported numbers may no longer match.",
                dest.name,
                expected,
                digest,
            )
    return dest


def download_all(cfg: Config | None = None, *, force: bool = False) -> dict[str, Path]:
    cfg = cfg or load_config()
    raw = cfg.resolve(cfg.data.raw_dir)
    out: dict[str, Path] = {}
    for name, sha in cfg.data.goemotions_files.items():
        out[name] = _fetch(f"{cfg.data.goemotions_base}/{name}", raw / name, sha, force)
    out["twitter_samples.zip"] = _fetch(
        cfg.data.twitter_url, raw / "twitter_samples.zip", cfg.data.twitter_sha256, force
    )
    log.info("%d files ready in %s", len(out), raw)
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    download_all()
