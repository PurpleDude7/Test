"""Command-line entry point: ``evalkit run suite.yaml --provider fake --report out.html``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, TextIO

import yaml

from evalkit import __version__
from evalkit.errors import EvalkitError
from evalkit.loader import load_suite
from evalkit.providers import create_provider, provider_registry
from evalkit.report import write_report
from evalkit.results import CaseResult, RunResult
from evalkit.runner import run_suite
from evalkit.scorers import scorer_registry

EXIT_OK = 0
EXIT_GATE_FAILED = 1
EXIT_USAGE = 2


def _option(raw: str) -> tuple[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    # Parse the value as YAML so `-O timeout_s=30` gives a number, `-O regex=true` a bool.
    return key, yaml.safe_load(value) if value else ""


def _fraction(raw: str) -> float:
    value = float(raw)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalkit", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evalkit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a suite and report results")
    run.add_argument("suite", help="path to the suite YAML file")
    run.add_argument("-p", "--provider", help="provider name (default: the suite's `provider`)")
    run.add_argument("-m", "--model", help="model override for the provider")
    run.add_argument(
        "-O",
        "--provider-option",
        dest="options",
        type=_option,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="provider config override; repeatable",
    )
    run.add_argument(
        "-r",
        "--report",
        action="append",
        default=[],
        metavar="PATH",
        help="write a report; .html or .json picks the format; repeatable",
    )
    run.add_argument(
        "-t", "--threshold", type=_fraction, help="pass-rate gate in [0, 1] (overrides suite)"
    )
    run.add_argument("-j", "--concurrency", type=int, default=4, help="parallel cases")
    run.add_argument(
        "--tag", dest="tags", action="append", default=[], help="only run cases with this tag"
    )
    run.add_argument("-q", "--quiet", action="store_true", help="print only the summary")

    validate = sub.add_parser("validate", help="check a suite file without running it")
    validate.add_argument("suite")

    sub.add_parser("list", help="list registered providers and scorers")
    return parser


def _print_case(case: CaseResult, out: TextIO) -> None:
    mark = "PASS" if case.passed else "FAIL"
    detail = case.error or "; ".join(f"{s.scorer}: {s.reason}" for s in case.scores if not s.passed)
    out.write(f"  {mark}  {case.case_id}" + (f"  - {detail}" if detail else "") + "\n")


def _print_summary(result: RunResult, out: TextIO) -> None:
    verdict = "PASSED" if result.passed else "FAILED"
    out.write(
        f"\n{result.suite}: {result.passed_count}/{result.total} passed "
        f"({result.pass_rate:.1%}), threshold {result.threshold:.1%} -> {verdict}\n"
    )


def _cmd_run(args: argparse.Namespace, out: TextIO) -> int:
    if args.concurrency < 1:
        raise EvalkitError("--concurrency must be >= 1")
    suite = load_suite(args.suite)
    try:
        suite = suite.filter_tags(args.tags)
    except ValueError as exc:
        raise EvalkitError(str(exc)) from exc
    name = args.provider or suite.provider
    if not name:
        raise EvalkitError("no provider: pass --provider or set `provider` in the suite")
    try:
        kind, config = suite.resolve_provider(name)
    except ValueError as exc:
        raise EvalkitError(str(exc)) from exc
    config.update(dict(args.options))
    if args.model:
        config["model"] = args.model
    provider = create_provider(kind, config)

    model = f" ({provider.model})" if provider.model else ""
    out.write(f"Running {suite.name!r}: {len(suite.cases)} cases on {name}{model}\n")
    try:
        result = run_suite(
            suite,
            provider,
            threshold=args.threshold,
            concurrency=args.concurrency,
            on_result=None if args.quiet else lambda c: _print_case(c, out),
        )
    finally:
        provider.close()

    _print_summary(result, out)
    for path in args.report:
        try:
            written = write_report(result, path)
        except (ValueError, OSError) as exc:
            raise EvalkitError(f"cannot write report {path}: {exc}") from exc
        out.write(f"Report written to {written}\n")
    return EXIT_OK if result.passed else EXIT_GATE_FAILED


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args, out)
        if args.command == "validate":
            suite = load_suite(args.suite)
            out.write(f"{args.suite}: OK ({len(suite.cases)} cases)\n")
            return EXIT_OK
        out.write("providers: " + ", ".join(provider_registry.names()) + "\n")
        out.write("scorers:   " + ", ".join(scorer_registry.names()) + "\n")
        return EXIT_OK
    except EvalkitError as exc:
        sys.stderr.write(f"evalkit: error: {exc}\n")
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
