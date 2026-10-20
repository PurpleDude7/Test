"""Command line: ``lectureiq analyze | ask | transcribe | serve``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

import yaml

from evalkit.errors import EvalkitError
from lectureiq import __version__
from lectureiq.audio import WhisperTranscriber, is_audio, transcribe_lecture
from lectureiq.engines import Engine, PackOptions
from lectureiq.export import flashcards_csv, to_markdown, to_vtt
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


def _transcription_args(parser: argparse.ArgumentParser, *, implied: bool = False) -> None:
    group = parser.add_argument_group("audio transcription (OpenAI-compatible endpoint)")
    if not implied:
        group.add_argument(
            "--transcribe",
            action="store_true",
            help="accept audio files and transcribe them (calls a paid API by default)",
        )
    group.add_argument("--transcribe-model", default="whisper-1", metavar="MODEL")
    group.add_argument("--transcribe-url", default="https://api.openai.com/v1", metavar="URL")
    group.add_argument(
        "--transcribe-key-env", default="OPENAI_API_KEY", metavar="VAR", help="API key variable"
    )
    group.add_argument("--language", help="spoken language as ISO-639-1, e.g. en (optional)")


def make_transcriber(args: argparse.Namespace) -> WhisperTranscriber | None:
    if not getattr(args, "transcribe", True):
        return None
    return WhisperTranscriber(
        model=args.transcribe_model,
        base_url=args.transcribe_url,
        api_key_env=args.transcribe_key_env,
        language=args.language,
    )


def _transcript_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "transcript",
        help=f"transcript (.txt/.md/.srt/.vtt) or, with --transcribe, audio (.mp3/.m4a/.wav/...); "
        f"'-' for stdin, {SAMPLE} for the demo",
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
    _transcription_args(an)

    qa = sub.add_parser("ask", help="answer a question from a transcript, with citations")
    _transcript_arg(qa)
    qa.add_argument("question")
    _engine_args(qa)
    _transcription_args(qa)

    tr = sub.add_parser("transcribe", help="turn a lecture recording into a WebVTT transcript")
    tr.add_argument("audio", help="audio file (.mp3/.m4a/.wav/.webm/...)")
    tr.add_argument("-o", "--out", metavar="PATH", help="output path (default: AUDIO.vtt)")
    _transcription_args(tr, implied=True)

    sv = sub.add_parser("serve", help="run the web app")
    _engine_args(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument(
        "--db",
        metavar="PATH",
        help="SQLite file to keep lectures across restarts (default: memory)",
    )
    _transcription_args(sv)
    return parser


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise EvalkitError(f"cannot read {path}: {exc.strerror or exc}") from exc


def _load(args: argparse.Namespace) -> Lecture:
    if args.transcript == SAMPLE:
        return parse_transcript(sample_transcript(), title=args.title or SAMPLE_TITLE)
    if args.transcript == "-":
        return parse_transcript(sys.stdin.read(), title=args.title or "Untitled lecture")
    path = Path(args.transcript)
    if is_audio(path.name):
        transcriber = make_transcriber(args)
        if transcriber is None:
            raise EvalkitError(f"{path.name} is audio; add --transcribe to transcribe it first")
        return transcribe_lecture(transcriber, _read_bytes(path), path.name, title=args.title)
    text = _read_bytes(path).decode("utf-8", errors="replace")
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


def _cmd_transcribe(args: argparse.Namespace, out: TextIO) -> None:
    path = Path(args.audio)
    transcriber = make_transcriber(args)
    assert transcriber is not None  # transcription is implied for this command
    lecture = transcribe_lecture(transcriber, _read_bytes(path), path.name)
    if lecture.has_timestamps:
        target, body = Path(args.out or path.with_suffix(".vtt")), to_vtt(lecture)
    else:  # the server returned plain text only
        target, body = Path(args.out or path.with_suffix(".txt")), lecture.text + "\n"
    try:
        target.write_text(body, encoding="utf-8")
    except OSError as exc:
        raise EvalkitError(f"cannot write {target}: {exc.strerror or exc}") from exc
    out.write(f"Transcribed {path.name}: {lecture.word_count:,} words -> {target}\n")


def _cmd_serve(args: argparse.Namespace, out: TextIO) -> None:  # pragma: no cover - blocking
    import uvicorn

    from lectureiq.store import SQLiteStore
    from lectureiq.web.app import create_app

    engine = _engine(args)
    store = SQLiteStore(args.db) if args.db else None
    app = create_app(
        engine, chunk_words=args.chunk_words, store=store, transcriber=make_transcriber(args)
    )
    extras = [f"db {args.db}" if args.db else "in-memory", "audio on" if args.transcribe else ""]
    details = ", ".join(e for e in extras if e)
    out.write(f"LectureIQ ({engine.name} engine, {details}) on http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port)


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    commands = {
        "analyze": _cmd_analyze,
        "ask": _cmd_ask,
        "transcribe": _cmd_transcribe,
        "serve": _cmd_serve,
    }
    try:
        commands[args.command](args, out)
    except EvalkitError as exc:
        sys.stderr.write(f"lectureiq: error: {exc}\n")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
