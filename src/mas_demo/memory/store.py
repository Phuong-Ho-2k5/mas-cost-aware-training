import sqlite3
import json
from pathlib import Path
from mas_demo.contracts import Intervention


def normalize_action(value):
    if value is None or value == 0:
        return None
    if isinstance(value, (float, int)):
        return Intervention(parameter="learning_rate", operation="multiply", value=value)
    return Intervention.model_validate(value)


def settings_context(dataset, signals, settings):
    normalized = {key: float(value) if value is not None else None for key, value in settings.items()}
    return f"{dataset}:smallcnn:{','.join(sorted(signals))}:" + json.dumps(normalized, sort_keys=True)


class MemoryStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS outcomes (
                context TEXT, factor REAL, status TEXT, gain REAL, seconds REAL,
                round_id INTEGER, PRIMARY KEY(context, factor, round_id))""")
            db.execute("""CREATE TABLE IF NOT EXISTS action_outcomes (
                context TEXT, action TEXT, status TEXT, gain REAL, seconds REAL,
                round_id INTEGER, PRIMARY KEY(context, action, round_id))""")
            for context, factor, status, gain, seconds, round_id in db.execute("SELECT * FROM outcomes").fetchall():
                action = normalize_action(factor)
                key = action.key if action else "null"
                db.execute("INSERT OR IGNORE INTO action_outcomes VALUES (?,?,?,?,?,?)",
                           (context, key, status, gain, seconds, round_id))

    def record(self, context, factor, status, gain, seconds, round_id):
        action = normalize_action(factor)
        key = action.key if action else "null"
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT OR REPLACE INTO action_outcomes VALUES (?,?,?,?,?,?)",
                       (context, key, status, gain, seconds, round_id))

    def migrate_legacy_contexts(self, events_path, dataset):
        """Copy old LR history into contexts reconstructed from each round's real evidence."""
        path = Path(events_path)
        if not path.exists():
            return 0
        evidence = None
        copied = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") == "evidence":
                evidence = event["evidence"]
            elif event.get("kind") == "diagnosis" and evidence and evidence.get("recent_events"):
                last = evidence["recent_events"][-1]
                if last.get("learning_rate") is None:
                    continue  # Keep history, but do not guess its training context.
                settings = {"learning_rate": last["learning_rate"],
                            "momentum": last.get("momentum") if last.get("momentum") is not None else 0.9,
                            "weight_decay": last.get("weight_decay") if last.get("weight_decay") is not None else 0.0,
                            "grad_clip_norm": last.get("grad_clip_norm")}
                old_context = f"{dataset}:smallcnn:{','.join(sorted(evidence['signals']))}"
                new_context = settings_context(dataset, evidence["signals"], settings)
                number = event["round"]
                for row in self.retrieve(old_context, number + 1):
                    if row["round_id"] == number:
                        self.record(new_context, row["intervention"], row["status"], row["gain"], row["seconds"], number)
                        copied += 1
        return copied

    def retrieve(self, context, before_round):
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute(
                "SELECT * FROM action_outcomes WHERE context=? AND round_id<? ORDER BY round_id, action",
                (context, before_round))]
            legacy = db.execute("SELECT * FROM outcomes WHERE context=? AND round_id<?", (context, before_round)).fetchall()
        # Legacy rows may have been written after opening the store (e.g. another old tool).
        seen = {(r["action"], r["round_id"]) for r in rows}
        for row in legacy:
            action = normalize_action(row["factor"])
            key = action.key if action else "null"
            if (key, row["round_id"]) not in seen:
                rows.append({"context": row["context"], "action": key, "status": row["status"],
                             "gain": row["gain"], "seconds": row["seconds"], "round_id": row["round_id"]})
        for row in rows:
            row["intervention"] = json.loads(row.pop("action"))
            item = row["intervention"]
            row["factor"] = item["value"] if item and item["parameter"] == "learning_rate" else None
        return sorted(rows, key=lambda r: (r["round_id"], json.dumps(r["intervention"], sort_keys=True)))
