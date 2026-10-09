"""Saved Reports library and audit log in SQLite (same file as the conversation checkpoints)."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Report:
    id: int
    title: str
    created_at: str


class ReportStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def save(self, user_id: str, thread_id: str, title: str, content: str) -> int:
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO reports (user_id, thread_id, title, content) VALUES (?, ?, ?, ?)",
                (user_id, thread_id, title, content),
            )
        return cur.lastrowid

    def find(self, user_id: str, keyword: str = "", thread_id: str | None = None) -> list[Report]:
        """The user's own reports, optionally only those mentioning `keyword` or made in one thread."""
        query = "SELECT id, title, created_at FROM reports WHERE user_id = ?"
        params: list = [user_id]
        if keyword:
            query += " AND instr(lower(title || ' ' || content), lower(?)) > 0"
            params.append(keyword)
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        return [Report(*row) for row in self.conn.execute(query + " ORDER BY id", params)]

    def delete(self, user_id: str, thread_id: str, reports: list[Report], request: str) -> int:
        """Deletes the given reports if they belong to the user. The deletion and its audit
        entry are one transaction: there is never a deletion without an audit record."""
        ids = [r.id for r in reports]
        details = {"request": request, "reports": [{"id": r.id, "title": r.title} for r in reports]}
        with self.conn:
            placeholders = ", ".join("?" * len(ids))
            deleted = self.conn.execute(
                f"DELETE FROM reports WHERE user_id = ? AND id IN ({placeholders})", [user_id, *ids]
            ).rowcount
            self.conn.execute(
                "INSERT INTO audit_log (user_id, thread_id, action, details) VALUES (?, ?, 'delete_reports', ?)",
                (user_id, thread_id, json.dumps({**details, "deleted": deleted})),
            )
        return deleted

    def close(self) -> None:
        self.conn.close()
