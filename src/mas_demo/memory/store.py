"""Canonical bundle history, independent of legacy one-parameter actions."""

import json
import sqlite3
from pathlib import Path


class MemoryStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS config_outcomes (
                context TEXT, base_hash TEXT, candidate_hash TEXT, round_id INTEGER,
                record TEXT, PRIMARY KEY(context, base_hash, candidate_hash, round_id))"""
            )

    def record(self, context, base_hash, candidate_hash, round_id, **fields):
        record = dict(
            context=context,
            base_config_hash=base_hash,
            config_hash=candidate_hash,
            round_id=round_id,
            **fields
        )
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO config_outcomes VALUES (?,?,?,?,?)",
                (
                    context,
                    base_hash,
                    candidate_hash,
                    round_id,
                    json.dumps(record, allow_nan=False),
                ),
            )

    def retrieve(self, context, before_round, limit=None):
        with sqlite3.connect(self.path) as db:
            rows = db.execute(
                "SELECT record FROM config_outcomes WHERE context=? AND round_id<? ORDER BY round_id DESC",
                (context, before_round),
            ).fetchall()
        records = [json.loads(row[0]) for row in rows]
        return records[:limit] if limit else records

    def check_cooldown(self, context, plan, round_id, cooldown):
        from mas_demo.configuration import ProposalError

        rows = self.retrieve(context, round_id)
        for candidate in plan.candidates:
            if any(
                row["base_config_hash"] == candidate.base_config_hash
                and row["config_hash"] == candidate.config_hash
                and row.get("status") in ("REJECT", "FAILED")
                and round_id - row["round_id"] <= cooldown
                for row in rows
            ):
                raise ProposalError(
                    "COOLDOWN",
                    "Candidate was recently rejected from this base",
                    proposal_id=candidate.proposal_id,
                )
