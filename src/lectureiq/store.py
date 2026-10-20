"""Where the web app keeps analyzed lectures: in memory, or in SQLite to survive restarts."""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from lectureiq.models import StudyPack
from lectureiq.transcript import Lecture, lecture_from_dict, lecture_to_dict


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Entry:
    id: str
    lecture: Lecture
    pack: StudyPack
    created_at: datetime = field(default_factory=_now)


@dataclass(frozen=True)
class Summary:
    """What the "recent lectures" list needs, without loading whole lectures."""

    id: str
    title: str
    created_at: datetime
    word_count: int
    engine: str


class Store(Protocol):
    def add(self, lecture: Lecture, pack: StudyPack) -> Entry: ...
    def get(self, lecture_id: str) -> Entry | None: ...
    def recent(self, limit: int = 20) -> list[Summary]: ...
    def delete(self, lecture_id: str) -> bool: ...


def _summary(entry: Entry) -> Summary:
    return Summary(
        id=entry.id,
        title=entry.pack.title,
        created_at=entry.created_at,
        word_count=entry.lecture.word_count,
        engine=entry.pack.engine,
    )


def _new_id() -> str:
    return secrets.token_urlsafe(8)


class MemoryStore:
    """Keeps the ``capacity`` most recently used lectures; older ones are evicted."""

    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, Entry] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, lecture: Lecture, pack: StudyPack) -> Entry:
        entry = Entry(id=_new_id(), lecture=lecture, pack=pack)
        with self._lock:
            self._items[entry.id] = entry
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return entry

    def get(self, lecture_id: str) -> Entry | None:
        with self._lock:
            entry = self._items.get(lecture_id)
            if entry is not None:
                self._items.move_to_end(lecture_id)
            return entry

    def recent(self, limit: int = 20) -> list[Summary]:
        with self._lock:
            entries = sorted(self._items.values(), key=lambda e: e.created_at, reverse=True)
        return [_summary(e) for e in entries[:limit]]

    def delete(self, lecture_id: str) -> bool:
        with self._lock:
            return self._items.pop(lecture_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS lectures (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    engine TEXT NOT NULL,
    lecture TEXT NOT NULL,
    pack TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS lectures_created_at ON lectures (created_at);
"""


class SQLiteStore:
    """One row per lecture: the parsed transcript and its study pack as JSON.

    A connection per operation keeps it safe to use from the server's thread pool; WAL
    mode lets reads proceed while a write is in progress.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10)) as conn, conn:
            yield conn

    def add(self, lecture: Lecture, pack: StudyPack) -> Entry:
        entry = Entry(id=_new_id(), lecture=lecture, pack=pack)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO lectures VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    pack.title,
                    entry.created_at.isoformat(),
                    lecture.word_count,
                    pack.engine,
                    json.dumps(lecture_to_dict(lecture)),
                    pack.model_dump_json(),
                ),
            )
        return entry

    def get(self, lecture_id: str) -> Entry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at, lecture, pack FROM lectures WHERE id = ?", (lecture_id,)
            ).fetchone()
        if row is None:
            return None
        return Entry(
            id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            lecture=lecture_from_dict(json.loads(row[2])),
            pack=StudyPack.model_validate_json(row[3]),
        )

    def recent(self, limit: int = 20) -> list[Summary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, created_at, word_count, engine FROM lectures "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            Summary(
                id=r[0],
                title=r[1],
                created_at=datetime.fromisoformat(r[2]),
                word_count=r[3],
                engine=r[4],
            )
            for r in rows
        ]

    def delete(self, lecture_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM lectures WHERE id = ?", (lecture_id,)).rowcount > 0
