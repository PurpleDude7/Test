"""Build the single-file LectureIQ demo page (published as a claude.ai artifact).

    python demo/build.py [OUTPUT]          # default: demo/dist/lectureiq.html

Inlines demo/lectureiq-core.js and the bundled sample lecture into demo/page.html.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_URL = "https://github.com/PurpleDude7/Test"


def build() -> str:
    page = (HERE / "page.html").read_text(encoding="utf-8")
    core = (HERE / "lectureiq-core.js").read_text(encoding="utf-8")
    sample_path = ROOT / "src" / "lectureiq" / "samples" / "intro_to_ml.vtt"
    sample = sample_path.read_text(encoding="utf-8")
    # Keep "</script>" out of the inlined data so it can't close the script tag early.
    sample_js = json.dumps(sample).replace("</", "<\\/")
    for marker in ("/*__CORE__*/", "__SAMPLE_VTT__", "__REPO_URL__"):
        if page.count(marker) != 1:
            raise SystemExit(f"page.html must contain {marker} exactly once")
    return (
        page.replace("/*__CORE__*/", core)
        .replace("__SAMPLE_VTT__", sample_js)
        .replace("__REPO_URL__", REPO_URL)
    )


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "dist" / "lectureiq.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
