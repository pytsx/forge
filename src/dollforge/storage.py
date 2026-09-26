"""Append-only metadata revisions and content-addressed local artifact storage."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar
from uuid import UUID

import structlog
from pydantic import BaseModel

from dollforge.domain.models import Artifact, Lineage, Provenance, Stage
from dollforge.errors import Conflict, NotFound

T = TypeVar("T", bound=BaseModel)
log = structlog.get_logger()


def canonical(value) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.blobs = self.root / "blobs"
        self.blobs.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.db_path = self.root / "metadata.sqlite3"
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS revisions (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS revision_lookup ON revisions(kind,id,seq);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, kind TEXT NOT NULL,
                    version INTEGER NOT NULL, body TEXT NOT NULL,
                    UNIQUE(project,kind,version));
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY, artifact_id TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        with self.lock:
            db = sqlite3.connect(self.db_path, timeout=30)
            try:
                with db:
                    yield db
            finally:
                db.close()

    def save(self, kind: str, identifier: UUID, value: BaseModel) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO revisions(kind,id,body) VALUES (?,?,?)",
                       (kind, str(identifier), canonical(value).decode()))

    def get(self, kind: str, identifier: UUID, model: type[T]) -> T:
        with self.connection() as db:
            row = db.execute("SELECT body FROM revisions WHERE kind=? AND id=? "
                             "ORDER BY seq DESC LIMIT 1", (kind, str(identifier))).fetchone()
        if not row:
            raise NotFound(f"{kind} não encontrado: {identifier}")
        return model.model_validate_json(row[0])

    def list(self, kind: str, model: type[T]) -> list[T]:
        with self.connection() as db:
            rows = db.execute("SELECT body FROM revisions WHERE seq IN "
                              "(SELECT MAX(seq) FROM revisions WHERE kind=? GROUP BY id) "
                              "ORDER BY seq DESC", (kind,)).fetchall()
        return [model.model_validate_json(row[0]) for row in rows]

    def put(self, payload: bytes, *, project_id: UUID, run_id: UUID, stage: Stage,
            kind: str, lineage: Lineage, provenance: Provenance,
            media_type: str = "application/json") -> Artifact:
        checksum = digest(payload)
        destination = self.blobs / checksum
        with self.connection() as db:
            for parent in lineage.input_artifact_ids:
                row = db.execute("SELECT project FROM artifacts WHERE id=?", (str(parent),)).fetchone()
                if not row or row[0] != str(project_id):
                    raise Conflict("Lineage deve referenciar artefatos existentes do mesmo projeto")
            if not destination.exists():
                with destination.open("xb") as stream:
                    stream.write(payload)
            elif digest(destination.read_bytes()) != checksum:
                raise Conflict("Integridade do armazenamento comprometida")
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM artifacts "
                                 "WHERE project=? AND kind=?", (str(project_id), kind)).fetchone()[0]
            artifact = Artifact(project_id=project_id, run_id=run_id, stage=stage, kind=kind,
                                version=version, sha256=checksum, size_bytes=len(payload),
                                lineage=lineage, provenance=provenance, media_type=media_type)
            db.execute("INSERT INTO artifacts VALUES (?,?,?,?,?)",
                       (str(artifact.artifact_id), str(project_id), kind, version,
                        artifact.model_dump_json()))
        log.info("artifact_created", artifact_id=str(artifact.artifact_id),
                 project_id=str(project_id), run_id=str(run_id), stage=stage, sha256=checksum)
        return artifact

    def metadata(self, identifier: UUID) -> Artifact:
        with self.connection() as db:
            row = db.execute("SELECT body FROM artifacts WHERE id=?", (str(identifier),)).fetchone()
        if not row:
            raise NotFound(f"Artefato não encontrado: {identifier}")
        return Artifact.model_validate_json(row[0])

    def read(self, identifier: UUID) -> bytes:
        artifact = self.metadata(identifier)
        data = (self.blobs / artifact.sha256).read_bytes()
        if digest(data) != artifact.sha256:
            raise Conflict(f"Checksum inválido: {identifier}")
        return data

    def json(self, identifier: UUID):
        return json.loads(self.read(identifier))

    def cached(self, key: str) -> Artifact | None:
        with self.connection() as db:
            row = db.execute("SELECT artifact_id FROM cache WHERE key=?", (key,)).fetchone()
        return self.metadata(UUID(row[0])) if row else None

    def set_cache(self, key: str, identifier: UUID) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO cache VALUES (?,?) ON CONFLICT(key) DO UPDATE "
                       "SET artifact_id=excluded.artifact_id", (key, str(identifier)))

    def artifacts(self, project_id: UUID) -> list[Artifact]:
        with self.connection() as db:
            rows = db.execute("SELECT body FROM artifacts WHERE project=? ORDER BY rowid DESC",
                              (str(project_id),)).fetchall()
        return [Artifact.model_validate_json(row[0]) for row in rows]
