"""The hosted demo runs a JavaScript port of the engine (demo/lectureiq-core.js).

These tests hold the port to the Python package: the same study pack on the sample
lecture, and the same evalkit QA suite passing. They need Node and skip without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from evalkit import FakeProvider, load_suite, run_suite
from lectureiq import analyze, parse_transcript
from lectureiq.offline import OfflineEngine
from lectureiq.sample import sample_transcript

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / "demo" / "lectureiq-core.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

SCRIPT = """
const L = require(process.argv[1]);
const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const lec = L.parseTranscript(input.transcript, input.title);
L.chunkLecture(lec);
const pack = L.offlineEngine.studyPack(lec, lec.chunks);
(async () => {
  const answers = {};
  for (const q of input.questions) answers[q] = await L.ask(lec, q, L.offlineEngine);
  process.stdout.write(JSON.stringify({ pack, answers, markdown: L.toMarkdown(pack) }));
})();
"""


def run_port(questions: list[str]) -> dict[str, Any]:
    assert NODE is not None
    payload = {"transcript": sample_transcript(), "title": "ML", "questions": questions}
    proc = subprocess.run(
        [NODE, "-e", SCRIPT, str(CORE)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result: dict[str, Any] = json.loads(proc.stdout)
    return result


def test_port_matches_python_study_pack() -> None:
    js = run_port([])["pack"]
    py = analyze(parse_transcript(sample_transcript(), title="ML"), OfflineEngine())
    assert js["tldr"] == py.tldr
    assert [s["heading"] for s in js["sections"]] == [s.heading for s in py.sections]
    assert [(t["term"], t["definition"], t["start"]) for t in js["terms"]] == [
        (t.term, t.definition, t.start) for t in py.terms
    ]
    assert [(k["text"], k["start"]) for k in js["key_points"]] == [
        (k.text, k.start) for k in py.key_points
    ]
    assert len(js["flashcards"]) == len(py.flashcards)
    assert len(js["quiz"]) == len(py.quiz)
    for q in js["quiz"]:  # options are shuffled differently, but must be well formed
        options = q["options"]
        assert 2 <= len(options) == len(set(options)) and 0 <= q["answer"] < len(options)


def test_port_passes_the_lectureiq_qa_suite() -> None:
    suite = load_suite(ROOT / "examples" / "lectureiq_qa.yaml")
    questions = [case.prompt_text() for case in suite.cases]
    answers = run_port(questions)["answers"]

    def reply(answer: dict[str, Any]) -> str:
        # Same "Sources:" convention as lectureiq.evalkit_provider.
        stamps = []
        for c in answer["citations"]:
            if c["start"] is not None:
                s = int(c["start"])
                stamps.append(f"{s // 60}:{s % 60:02d}")
        text: str = answer["answer"]
        return text + ("\n\nSources: " + ", ".join(dict.fromkeys(stamps)) if stamps else "")

    canned = FakeProvider.with_responses({q: reply(answers[q]) for q in questions})
    result = run_suite(suite, canned)
    failures = [(c.case_id, c.output) for c in result.cases if not c.passed]
    assert failures == []
