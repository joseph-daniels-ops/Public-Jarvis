"""Memória episódica e gerenciamento de tarefas para o Public-Jarvis.

Implementa vetorização determinística baseada em hashlib.blake2b, cálculo de similaridade
de cosseno, armazenamento SQLite e recuperação híbrida (cosseno + overlap lexical).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Representação de um registro de memória armazenado."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    content: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    score: float = 0.0


class TaskItem(BaseModel):
    """Representação de uma tarefa enfileirada no Jarvis."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str
    intent: str
    status: str = "pending"
    result: str | None = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def deterministic_vectorize(text: str, dim: int = 128) -> list[float]:
    """Gera um vetor normalizado L2 determinístico para o texto usando hashlib.blake2b.

    Não utiliza o hash() nativo do Python (sujeito à randomização por processo via PYTHONHASHSEED).
    """
    if dim <= 0:
        raise ValueError("dim deve ser um inteiro positivo maior que zero")

    vec = [0.0] * dim
    tokens = re.findall(r"\w+", text.lower())
    if not tokens and text.strip():
        tokens = [text.strip().lower()]

    if not tokens:
        return vec

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, byteorder="little") % dim
        vec[idx] += 1.0

    # Normalização L2
    norm = math.sqrt(sum(val * val for val in vec))
    if norm > 0.0:
        vec = [val / norm for val in vec]
    return vec


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calcula a similaridade de cosseno entre dois vetores numéricos."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0

    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    val = dot / (norm1 * norm2)
    return max(-1.0, min(1.0, val))


class JarvisMemoryStore:
    """Armazenamento SQLite para memórias episódicas e fila de tarefas."""

    def __init__(self, db_path: Path | str = ":memory:"):
        self.db_path = db_path
        if isinstance(db_path, str) and db_path != ":memory:":
            path_obj = Path(db_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            self._db_str = str(path_obj.resolve())
        elif isinstance(db_path, Path):
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db_str = str(db_path.resolve())
        else:
            self._db_str = ":memory:"

        self._conn = sqlite3.connect(self._db_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Cria as tabelas e índices necessários no SQLite."""
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    entry_id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at)"
            )

    def add_memory(self, topic: str, content: str) -> MemoryEntry:
        """Adiciona uma nova memória e indexa seu vetor determinístico."""
        entry = MemoryEntry(topic=topic, content=content, score=0.0)
        vec = deterministic_vectorize(f"{topic} {content}")
        vec_json = json.dumps(vec)

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memories (entry_id, topic, content, created_at, vector_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    entry.entry_id,
                    entry.topic,
                    entry.content,
                    entry.created_at,
                    vec_json,
                ),
            )
        return entry

    def search_memories(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Busca memórias combinando similaridade de cosseno com sobreposição lexical."""
        if top_k <= 0:
            return []

        q_vec = deterministic_vectorize(query)
        q_tokens = set(re.findall(r"\w+", query.lower()))

        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT entry_id, topic, content, created_at, vector_json FROM memories"
        )
        rows = cursor.fetchall()
        if not rows:
            return []

        scored: list[tuple[float, MemoryEntry]] = []
        for row in rows:
            try:
                vec = json.loads(row["vector_json"])
            except Exception:
                vec = deterministic_vectorize(f"{row['topic']} {row['content']}")

            cos_sim = cosine_similarity(q_vec, vec)

            doc_tokens = set(
                re.findall(r"\w+", f"{row['topic']} {row['content']}".lower())
            )
            if q_tokens:
                overlap = len(q_tokens.intersection(doc_tokens)) / len(q_tokens)
            else:
                overlap = 0.0

            # 50% cosseno vetorial + 50% overlap lexical
            combined_score = 0.5 * max(0.0, cos_sim) + 0.5 * overlap

            entry = MemoryEntry(
                entry_id=row["entry_id"],
                topic=row["topic"],
                content=row["content"],
                created_at=row["created_at"],
                score=round(combined_score, 4),
            )
            scored.append((combined_score, entry))

        # Ordenação estável por score decrescente, desempate por created_at mais recente
        scored.sort(key=lambda item: (item[0], item[1].created_at), reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def enqueue_task(self, prompt: str, intent: str) -> TaskItem:
        """Insere uma nova tarefa na fila com status inicial 'pending'."""
        task = TaskItem(
            prompt=prompt,
            intent=intent,
            status="pending",
            result=None,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO tasks (task_id, prompt, intent, status, result, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.prompt,
                    task.intent,
                    task.status,
                    task.result,
                    task.created_at,
                ),
            )
        return task

    def complete_task(self, task_id: str, status: str, result: str) -> None:
        """Atualiza o status e resultado de uma tarefa existente."""
        with self._conn:
            self._conn.execute(
                "UPDATE tasks SET status = ?, result = ? WHERE task_id = ?",
                (status, result, task_id),
            )

    def list_tasks(self, limit: int = 20) -> list[TaskItem]:
        """Lista as tarefas mais recentes ordenadas por created_at decrescente."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT task_id, prompt, intent, status, result, created_at
            FROM tasks
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            TaskItem(
                task_id=row["task_id"],
                prompt=row["prompt"],
                intent=row["intent"],
                status=row["status"],
                result=row["result"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def close(self) -> None:
        """Fecha a conexão ativa com o banco SQLite."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> JarvisMemoryStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
