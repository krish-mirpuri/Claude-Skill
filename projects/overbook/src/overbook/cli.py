"""Command line entry point: ``overbook <stage>``."""

from __future__ import annotations

import argparse
import logging
import sys

from overbook.config import DEFAULT_CONFIG_PATH, load_config

STAGES = ("download", "prepare", "train", "backtest", "report", "all", "serving-context")


def _setup_logging(verbosity: int) -> None:
    level = logging.DEBUG if verbosity > 1 else logging.INFO if verbosity else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="overbook",
        description="Calibrated cancellation risk and revenue-optimal overbooking limits.",
    )
    p.add_argument("stage", choices=STAGES, help="pipeline stage to run")
    p.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="config YAML")
    p.add_argument(
        "-v", "--verbose", action="count", default=1, help="-v for info (default), -vv for debug"
    )
    p.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    p.add_argument(
        "--force", action="store_true", help="re-download the raw extract even if present"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(0 if args.quiet else args.verbose)
    cfg = load_config(args.config)

    from overbook import pipeline
    from overbook.data.download import download

    if args.stage == "download":
        download(cfg, force=args.force)
    elif args.stage == "prepare":
        download(cfg)
        pipeline.run_prepare(cfg)
    elif args.stage == "train":
        pipeline.run_train(cfg)
    elif args.stage == "backtest":
        pipeline.run_backtest(cfg)
    elif args.stage == "report":
        pipeline.run_report(cfg)
    elif args.stage == "serving-context":
        from overbook.api.context import build_serving_context

        build_serving_context(cfg)
    elif args.stage == "all":
        download(cfg)
        pipeline.run_all(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
