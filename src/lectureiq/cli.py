"""Command line: ``lectureiq analyze``, ``lectureiq ask`` and ``lectureiq serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

import yaml

from evalkit.errors import EvalkitError
from lectureiq import __version__
from lectureiq.engines import Engine, PackOptions
from lectureiq.export import flashcards_csv, to_markdown
from lectureiq.models import StudyPack
from lectureiq.pipeline import DEFAULT_CHUNK_WORDS, analyze, ask, make_engine
from lectureiq.sample import SAMPLE_TITLE, sample_transcript
from lectureiq.transcript import Lecture, format_ts, parse_transcript

SAMPLE = "@sample"


def _option(raw: str) -> tuple[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {raw!r}")
    return key, yaml.safe_load(value) if value else ""


def _engine_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("engine")
    group.add_argument(
        "-p",
        "--provider",
        default="offline",
        help="'offline' (no LLM, default) or an evalkit provider: openai, anthropic, ...",
    )
    group.add_argument("-m", "--model", help="model name for the provider")
    group.add_argument(
        "-O",
        "--provider-option",
        dest="options",
        type=_option,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="provider config, e.g. -O base_url=http://localhost:11434/v1",
    )
    group.add_argument("--cache", metavar="DIR", help="reuse identical model replies from DIR")
    group.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS)


def _transcript_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "transcript",
        help=f"transcript file (.txt/.md/.srt/.vtt), '-' for stdin, {SAMPLE} for the demo",
    )
    parser.add_argument("-t", "--title", help="lecture title (default: from the file name)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lectureiq", description="Turn lecture transcripts into study material."
    )
    parser.add_argument("--version", action="version", version=f"lectureiq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    an = sub.add_parser("analyze", help="build a study pack from a transcript")
    _transcript_arg(an)
    _engine_args(an)
    an.add_argument("--json", metavar="PATH", help="write the full study pack as JSON")
    an.add_argument("--markdown", metavar="PATH", help="write a Markdown study guide")
    an.add_argument("--csv", metavar="PATH", help="write flashcards as CSV (Anki import)")

    qa = sub.add_parser("ask", help="answer a question from a transcript, with citations")
    _transcript_arg(qa)
    qa.add_argument("question")
    _engine_args(qa)

    sv = sub.add_parser("serve", help="run the web app")
    _engine_args(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    return parser


def _load(args: argparse.Namespace) -> Lecture:
    if args.transcript == SAMPLE:
        return parse_transcript(sample_transcript(), title=args.title or SAMPLE_TITLE)
    if args.transcript == "-":
        return parse_transcript(sys.stdin.read(), title=args.title or "Untitled lecture")
    path = Path(args.transcript)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EvalkitError(f"cannot read {path}: {exc.strerror or exc}") from exc
    return parse_transcript(text, title=args.title or path.stem.replace("_", " "))


def _engine(args: argparse.Namespace) -> Engine:
    config: dict[str, Any] = dict(args.options)
    if args.model:
        config["model"] = args.model
    if args.provider == "offline" and (config or args.cache):
        raise EvalkitError("--model, -O and --cache only apply to LLM providers")
    return make_engine(args.provider, config, cache_dir=args.cache)


def _print_pack(pack: StudyPack, out: TextIO) -> None:
    out.write(f"\n{pack.title}\n{'=' * len(pack.title)}\n\n{pack.tldr}\n")
    if pack.sections:
        out.write("\nOutline\n")
        for i, s in enumerate(pack.sections, 1):
            at = f" [{format_ts(s.start)}]" if s.start is not None else ""
            out.write(f"  {i}. {s.heading}{at}\n")
    out.write(
        f"\n{len(pack.key_points)} key points · {len(pack.terms)} terms · "
        f"{len(pack.flashcards)} flashcards · {len(pack.quiz)} quiz questions "
        f"({pack.engine} engine)\n"
    )
    for warning in pack.warnings:
        out.write(f"warning: {warning}\n")


def _cmd_analyze(args: argparse.Namespace, out: TextIO) -> None:
    lecture = _load(args)
    pack = analyze(lecture, _engine(args), options=PackOptions(), chunk_words=args.chunk_words)
    _print_pack(pack, out)
    outputs: list[tuple[str | None, Callable[[], str]]] = [
        (args.json, lambda: pack.model_dump_json(indent=2) + "\n"),
        (args.markdown, lambda: to_markdown(pack)),
        (args.csv, lambda: flashcards_csv(pack)),
    ]
    for path, render in outputs:
        if path:
            try:
                Path(path).write_text(render(), encoding="utf-8")
            except OSError as exc:
                raise EvalkitError(f"cannot write {path}: {exc.strerror or exc}") from exc
            out.write(f"Wrote {path}\n")


def _cmd_ask(args: argparse.Namespace, out: TextIO) -> None:
    lecture = _load(args)
    try:
        answer = ask(lecture, args.question, _engine(args), chunk_words=args.chunk_words)
    except ValueError as exc:
        raise EvalkitError(str(exc)) from exc
    out.write(answer.answer + "\n")
    for c in answer.citations:
        where = f"part {c.chunk_id}" + (f" at {format_ts(c.start)}" if c.start is not None else "")
        out.write(f"  ↳ {where}" + (f": “{c.quote}”" if c.quote else "") + "\n")


def _cmd_serve(args: argparse.Namespace, out: TextIO) -> None:  # pragma: no cover - blocking
    import uvicorn

    from lectureiq.web.app import create_app

    engine = _engine(args)
    out.write(f"LectureIQ ({engine.name} engine) on http://{args.host}:{args.port}\n")
    uvicorn.run(create_app(engine, chunk_words=args.chunk_words), host=args.host, port=args.port)


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    commands = {"analyze": _cmd_analyze, "ask": _cmd_ask, "serve": _cmd_serve}
    try:
        commands[args.command](args, out)
    except EvalkitError as exc:
        sys.stderr.write(f"lectureiq: error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
