"""EmotWen Synthetic Dataset Reviewer -- FastAPI backend."""

from __future__ import annotations

import json
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DB_PATH = Path(__file__).parent / "feedback.db"

SCHEMA = """\
CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_idx INTEGER NOT NULL UNIQUE,
    rating          TEXT NOT NULL CHECK(rating IN ('good', 'okay', 'bad')),
    tags            TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_feedback_conv ON feedback(conversation_idx);
"""


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.close()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def upsert_feedback(conn: sqlite3.Connection, conversation_idx: int, rating: str, tags: list[str]) -> dict:
    tags_json = json.dumps(tags)
    conn.execute(
        """INSERT INTO feedback (conversation_idx, rating, tags)
           VALUES (?, ?, ?)
           ON CONFLICT(conversation_idx) DO UPDATE SET
               rating = excluded.rating,
               tags = excluded.tags,
               updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
        """,
        (conversation_idx, rating, tags_json),
    )
    conn.commit()
    row = conn.execute(
        "SELECT conversation_idx, rating, tags, created_at, updated_at FROM feedback WHERE conversation_idx = ?",
        (conversation_idx,),
    ).fetchone()
    return _row_to_feedback(row)


def get_feedback_by_idx(conn: sqlite3.Connection, idx: int) -> dict | None:
    row = conn.execute(
        "SELECT conversation_idx, rating, tags, created_at, updated_at FROM feedback WHERE conversation_idx = ?",
        (idx,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_feedback(row)


def get_feedback_map(conn: sqlite3.Connection, indices: list[int]) -> dict[int, dict]:
    if not indices:
        return {}
    placeholders = ",".join("?" for _ in indices)
    rows = conn.execute(
        f"SELECT conversation_idx, rating, tags FROM feedback WHERE conversation_idx IN ({placeholders})",
        indices,
    ).fetchall()
    result: dict[int, dict] = {}
    for row in rows:
        result[row["conversation_idx"]] = {
            "rating": row["rating"],
            "tags": json.loads(row["tags"]),
        }
    return result


def get_all_reviewed_indices(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT conversation_idx FROM feedback").fetchall()
    return {row["conversation_idx"] for row in rows}


def _row_to_feedback(row: sqlite3.Row) -> dict:
    return {
        "conversation_idx": row["conversation_idx"],
        "rating": row["rating"],
        "tags": json.loads(row["tags"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# ---------------------------------------------------------------------------
# Dataset state
# ---------------------------------------------------------------------------

dataset = None
dataset_ready = False
source_list: list[str] = []
source_index: dict[str, list[int]] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dataset, dataset_ready, source_list, source_index

    init_db()

    from datasets import load_dataset
    dataset = load_dataset("brianist/emotwen-3.5-synthetic", split="train")

    all_sources = dataset["source"]
    source_list = sorted(set(all_sources))
    source_index = {}
    for s in source_list:
        source_index[s] = []
    for i, s in enumerate(all_sources):
        source_index[s].append(i)

    dataset_ready = True
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_dataset():
    if not dataset_ready:
        raise HTTPException(status_code=503, detail="Dataset still loading")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

VALID_TAGS = {"advice-leaking", "too-long", "off-topic", "repetitive"}


class FeedbackRequest(BaseModel):
    conversation_idx: int
    rating: Literal["good", "okay", "bad"]
    tags: list[Literal["advice-leaking", "too-long", "off-topic", "repetitive"]] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preview(messages: list[dict]) -> str:
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if len(content) > 80:
                return content[:80] + "..."
            return content
    for msg in messages:
        if msg["role"] != "system":
            content = msg["content"]
            if len(content) > 80:
                return content[:80] + "..."
            return content
    return "(no preview)"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/sources")
def get_sources(_=Depends(require_dataset)):
    return {"sources": source_list}


@app.get("/api/conversations")
def list_conversations(
    source: str | None = None,
    status: Literal["reviewed", "unreviewed"] | None = None,
    offset: int = 0,
    limit: int = Query(default=50, le=10000),
    _=Depends(require_dataset),
):
    # Build candidate indices
    if source is not None:
        if source not in source_index:
            return {"total": 0, "offset": offset, "limit": limit, "items": []}
        candidates = source_index[source]
    else:
        candidates = list(range(len(dataset)))

    # Status filter
    conn = get_db()
    try:
        if status is not None:
            reviewed = get_all_reviewed_indices(conn)
            if status == "reviewed":
                candidates = [i for i in candidates if i in reviewed]
            else:
                candidates = [i for i in candidates if i not in reviewed]

        total = len(candidates)
        page = candidates[offset : offset + limit]

        # Get feedback for the page
        fb_map = get_feedback_map(conn, page)
    finally:
        conn.close()

    items = []
    for idx in page:
        row = dataset[idx]
        fb = fb_map.get(idx)
        items.append({
            "idx": idx,
            "source": row["source"],
            "emotion_label": row["emotion_label"],
            "n_turns": row["n_turns"],
            "preview": _preview(row["messages"]),
            "rating": fb["rating"] if fb else None,
            "tags": fb["tags"] if fb else [],
        })

    return {"total": total, "offset": offset, "limit": limit, "items": items}


@app.get("/api/conversations/{idx}")
def get_conversation(idx: int, _=Depends(require_dataset)):
    if idx < 0 or idx >= len(dataset):
        raise HTTPException(status_code=404, detail=f"Conversation index {idx} out of range")

    row = dataset[idx]
    conn = get_db()
    try:
        fb = get_feedback_by_idx(conn, idx)
    finally:
        conn.close()

    return {
        "idx": idx,
        "source": row["source"],
        "emotion_label": row["emotion_label"],
        "messages": row["messages"],
        "feedback": fb if fb else None,
    }


@app.post("/api/feedback")
def post_feedback(req: FeedbackRequest, _=Depends(require_dataset)):
    if req.conversation_idx < 0 or req.conversation_idx >= len(dataset):
        raise HTTPException(status_code=422, detail=f"Conversation index {req.conversation_idx} out of range")

    conn = get_db()
    try:
        result = upsert_feedback(conn, req.conversation_idx, req.rating, req.tags)
    finally:
        conn.close()

    return result


@app.get("/api/stats")
def get_stats(source: str | None = None, _=Depends(require_dataset)):
    conn = get_db()
    try:
        if source is not None:
            if source not in source_index:
                return {
                    "total": 0,
                    "reviewed": 0,
                    "ratings": {"good": 0, "okay": 0, "bad": 0},
                    "tags": {"advice-leaking": 0, "too-long": 0, "off-topic": 0, "repetitive": 0},
                }
            indices = source_index[source]
            total = len(indices)
            placeholders = ",".join("?" for _ in indices)
            rows = conn.execute(
                f"SELECT rating, tags FROM feedback WHERE conversation_idx IN ({placeholders})",
                indices,
            ).fetchall()
        else:
            total = len(dataset)
            rows = conn.execute("SELECT rating, tags FROM feedback").fetchall()

        reviewed = len(rows)
        ratings = {"good": 0, "okay": 0, "bad": 0}
        tags_count = {"advice-leaking": 0, "too-long": 0, "off-topic": 0, "repetitive": 0}

        for row in rows:
            ratings[row["rating"]] += 1
            for tag in json.loads(row["tags"]):
                if tag in tags_count:
                    tags_count[tag] += 1

        result: dict = {
            "total": total,
            "reviewed": reviewed,
            "ratings": ratings,
            "tags": tags_count,
        }

        if source is None:
            by_source: dict = {}
            all_reviewed = get_all_reviewed_indices(conn)
            for s in source_list:
                s_indices = source_index[s]
                by_source[s] = {
                    "total": len(s_indices),
                    "reviewed": len(all_reviewed & set(s_indices)),
                }
            result["by_source"] = by_source

        return result
    finally:
        conn.close()
