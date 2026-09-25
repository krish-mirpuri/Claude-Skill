"""Command line entry point: ``nuance <stage>``."""

from __future__ import annotations

import argparse
import logging
import sys

from nuance.config import DEFAULT_CONFIG_PATH, load_config

STAGES = ("download", "train", "audit", "transfer", "active", "report", "all")


def _setup_logging(verbose: int, quiet: bool) -> None:
    level = logging.WARNING if quiet else (logging.DEBUG if verbose > 1 else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nuance",
        description="Multi-label emotion classification, scored honestly.",
    )
    p.add_argument("stage", choices=STAGES)
    p.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("-v", "--verbose", action="count", default=1)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--force", action="store_true", help="re-download raw corpora")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose, args.quiet)
    cfg = load_config(args.config)

    from nuance import pipeline
    from nuance.data.download import download_all

    if args.stage == "download":
        download_all(cfg, force=args.force)
    elif args.stage == "all":
        download_all(cfg)
        pipeline.run_all(cfg)
    else:
        download_all(cfg)
        {
            "train": pipeline.run_train,
            "audit": pipeline.run_audit,
            "transfer": pipeline.run_transfer,
            "active": pipeline.run_active,
            "report": pipeline.run_report,
        }[args.stage](cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
