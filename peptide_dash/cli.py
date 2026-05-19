# peptide_dash/cli.py
from __future__ import annotations

import argparse

from .app import create_app
from .data.context import DataContext
from .data.loader import BackgroundLoader


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Peptide Feature Explorer (modular wrapper)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--mode",
        choices=["modular"],
        default="modular",
        help="Reserved for future modes; currently only 'modular' is supported.",
    )
    ap.add_argument(
        "--data-dir",
        default=None,
        help="Path to GLOBAL_DATA or its parent directory.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--timeseries-dir",
        default=None,
        help="Folder containing $VARIANT/METRIC_REP#.xvg files.",
    )

    # Sampling controls (mutually exclusive)
    sampling = ap.add_mutually_exclusive_group()
    sampling.add_argument(
        "--sample5",
        action="store_true",
        help="Quick test: load ~5%% of VARIANTS (group-wise sampling).",
    )
    sampling.add_argument(
        "--sample-frac",
        dest="sample_frac",
        type=float,
        default=None,
        help="Sample this fraction of variants (0–1].",
    )

    ap.add_argument(
        "--sample-seed",
        dest="sample_seed",
        type=int,
        default=0,
        help="Random seed for variant sampling.",
    )
    ap.add_argument(
        "--sample-key",
        dest="sample_key",
        default=None,
        help="Reserved for future keyed sampling strategies.",
    )

    # Dev / performance tuning
    ap.add_argument(
        "--dev-quick",
        action="store_true",
        help="Quick startup: sniff a small numeric subset of columns.",
    )
    ap.add_argument(
        "--dev-cache",
        action="store_true",
        help="Enable fastboot cache for feature subsets.",
    )
    ap.add_argument(
        "--dev-mock",
        action="store_true",
        help="Build an empty mock DataContext (for layout testing).",
    )
    ap.add_argument(
        "--dev-max-files",
        dest="dev_max_files",
        type=int,
        default=None,
        help="Maximum number of feature files to scan/load.",
    )
    ap.add_argument(
        "--dev-cols",
        nargs="+",
        default=None,
        help="Explicit list of feature columns to load.",
    )
    ap.add_argument(
        "--dev-cache-dir",
        dest="dev_cache_dir",
        default=None,
        help="Directory for fastboot cache files (parquet).",
    )
    ap.add_argument(
        "--sync",
        action="store_true",
        help=(
            "Load data synchronously before starting the server "
            "(old behaviour; no loading screen)."
        ),
    )

    return ap


def _sampling_fraction_or_error(
    ap: argparse.ArgumentParser, args: argparse.Namespace
) -> float | None:
    frac = 0.05 if args.sample5 else args.sample_frac
    if frac is None:
        return None
    if not (0.0 < float(frac) <= 1.0):
        ap.error("--sample-frac must be in the range (0, 1].")
    return float(frac)


def main(argv: list[str] | None = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)

    frac = _sampling_fraction_or_error(ap, args)

    # ------------------------------------------------------------------
    # dev-mock: no data at all, just test the layout
    # ------------------------------------------------------------------
    if args.dev_mock:
        ctx = DataContext.from_cli(
            data_dir=args.data_dir,
            sample5=args.sample5,
            sample_frac=frac,
            sample_seed=args.sample_seed,
            sample_key=args.sample_key,
            dev_quick=args.dev_quick,
            dev_cache=args.dev_cache,
            dev_mock=True,
            dev_max_files=args.dev_max_files,
            dev_cols=args.dev_cols,
            dev_cache_dir=args.dev_cache_dir,
            timeseries_dir=args.timeseries_dir,
        )
        app, server = create_app(ctx, loader=None)
        print(f"\n  Peptide Dash (mock) → http://{args.host}:{args.port}/\n")
        app.run(host=args.host, port=int(args.port), debug=args.debug)
        return

    # ------------------------------------------------------------------
    # Synchronous path (--sync flag): old behaviour, blocks before start
    # ------------------------------------------------------------------
    if args.sync:
        ctx = DataContext.from_cli(
            data_dir=args.data_dir,
            sample5=args.sample5,
            sample_frac=frac,
            sample_seed=args.sample_seed,
            sample_key=args.sample_key,
            dev_quick=args.dev_quick,
            dev_cache=args.dev_cache,
            dev_mock=False,
            dev_max_files=args.dev_max_files,
            dev_cols=args.dev_cols,
            dev_cache_dir=args.dev_cache_dir,
            timeseries_dir=args.timeseries_dir,
        )
        if args.timeseries_dir is not None:
            ctx.timeseries_dir = args.timeseries_dir
        app, server = create_app(ctx, loader=None)
        print(f"\n  Peptide Dash → http://{args.host}:{args.port}/\n")
        app.run(host=args.host, port=int(args.port), debug=args.debug)
        return

    # ------------------------------------------------------------------
    # Async path (default): start server immediately, load in background
    # ------------------------------------------------------------------
    from pathlib import Path
    from .data import io

    # Resolve data dir early so DataContext.__post_init__ doesn't fail
    data_dir_raw = args.data_dir or str(Path.cwd())
    base_dir, _ = io._resolve_layout(data_dir_raw)
    data_dir_resolved = str(base_dir)

    # Create an *empty* context – the server can start with this
    ctx = DataContext.empty(
        data_dir=data_dir_resolved,
        timeseries_dir=args.timeseries_dir,
    )

    # Collect loader kwargs
    cli_kwargs = dict(
        sample_frac=frac,
        sample_seed=args.sample_seed,
        dev_quick=args.dev_quick,
        dev_cache=args.dev_cache,
        dev_max_files=args.dev_max_files,
        dev_cols=args.dev_cols,
        dev_cache_dir=args.dev_cache_dir,
    )

    loader = BackgroundLoader(ctx, cli_kwargs)
    loader.start()

    app, server = create_app(ctx, loader=loader)

    print(
        f"\n  Peptide Dash → http://{args.host}:{args.port}/\n"
        "  Data is loading in the background – open the URL now.\n"
    )
    app.run(host=args.host, port=int(args.port), debug=args.debug)


if __name__ == "__main__":
    main()
