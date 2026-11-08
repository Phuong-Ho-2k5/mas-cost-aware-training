import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel


def json_default(obj):
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, default=json_default, ensure_ascii=False,
                              allow_nan=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def append_event(path, kind, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema_version": 1, "time": datetime.now(timezone.utc).isoformat(),
              "kind": kind, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, default=json_default, ensure_ascii=False,
                                allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
