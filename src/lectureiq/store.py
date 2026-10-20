"""In-memory lecture store for the web app (bounded, thread-safe)."""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass

from lectureiq.models import StudyPack
from lectureiq.transcript import Lecture


@dataclass
class Entry:
    id: str
    lecture: Lecture
    pack: StudyPack


class LectureStore:
    """Keeps the ``capacity`` most recently used lectures; older ones are evicted."""

    def __init__(self, capacity: int = 50) -> None:
        self.capacity = capacity
        self._items: OrderedDict[str, Entry] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, lecture: Lecture, pack: StudyPack) -> Entry:
        entry = Entry(id=secrets.token_urlsafe(8), lecture=lecture, pack=pack)
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

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
