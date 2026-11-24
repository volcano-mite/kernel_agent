"""Filesystem run records shared by the server and the host CLI."""

import fcntl
import hashlib
import json
import os
import re
import secrets
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path):
    with path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def create(self, config):
        run_id = uuid.uuid4().hex
        token = secrets.token_urlsafe(32)
        path = self.path(run_id)
        path.mkdir(mode=0o700, parents=True)
        for directory in ("candidates", "evaluations", "diagnostics", "acceptance"):
            (path / directory).mkdir()
        write_json(path / "run.json", {
            **config,
            "schema_version": 1,
            "run_id": run_id,
            "created_at": timestamp(),
            "token_sha256": self.token_digest(token),
        })
        return run_id, token

    def path(self, run_id):
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise FileNotFoundError("Unknown run")
        return self.root / run_id

    def config(self, run_id):
        return read_json(self.path(run_id) / "run.json")

    @staticmethod
    def token_digest(token):
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def authenticate(self, run_id, token):
        try:
            digest = self.config(run_id)["token_sha256"]
        except FileNotFoundError:
            return False
        return secrets.compare_digest(digest, self.token_digest(token))

    @staticmethod
    def snapshot(run, source):
        content = source.encode("utf-8")
        candidate_id = hashlib.sha256(content).hexdigest()
        candidates = run / "candidates"
        path = candidates / f"{candidate_id}.cu"
        if not path.exists():
            with NamedTemporaryFile(dir=candidates, delete=False) as file:
                temporary = Path(file.name)
                try:
                    file.write(content)
                    file.flush()
                    os.fchmod(file.fileno(), 0o400)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return candidate_id

    @staticmethod
    def source(run, candidate_id):
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_id):
            raise FileNotFoundError("Unknown candidate")
        source = (run / "candidates" / f"{candidate_id}.cu").read_bytes().decode("utf-8")
        if hashlib.sha256(source.encode("utf-8")).hexdigest() != candidate_id:
            raise ValueError("Candidate snapshot hash mismatch")
        return source

    @staticmethod
    def records(directory):
        return [read_json(path) for path in directory.glob("*.json")]
