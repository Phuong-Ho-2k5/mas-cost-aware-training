import json
import pytest
from pydantic import ValidationError
from mas_demo.contracts import TaskSpec, TrainingConfig
from mas_demo.cli import load_spec, migrate_legacy_config
from mas_demo.memory.store import MemoryStore


def test_new_config_has_no_implicit_action_or_horizon_defaults(tmp_path):
    config = tmp_path / "new.yaml"
    config.write_text(
        "schema_version: 3\ntraining:\n  learning_rate: 0.027\npolicy:\n  max_trials: 3\n"
    )
    spec = load_spec(config)
    assert spec.training.learning_rate == 0.027
    dumped = spec.model_dump()
    assert not {"allowed_interventions", "lr_factors", "trial_horizon"} & dumped.keys()
    with pytest.raises(ValidationError):
        TaskSpec(allowed_interventions=[])


def test_explicit_migration_drops_menus_but_preserves_initial_settings():
    legacy = {
        "learning_rate": 0.1,
        "momentum": 0.99,
        "max_epochs": 100,
        "allowed_interventions": [],
        "lr_factors": [0.1],
        "trial_horizon": 2,
    }
    spec = migrate_legacy_config(legacy)
    assert spec.training.learning_rate == 0.1
    assert spec.policy.max_epochs == 100
    assert spec.observation_interval == 1 and spec.window == 5
    assert (
        not {"allowed_interventions", "lr_factors", "trial_horizon"}
        & spec.model_dump().keys()
    )


def test_bundle_history_records_horizon_and_distinguishes_base_configs(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.record(
        "dataset",
        "base-a",
        "candidate",
        1,
        diff={"learning_rate": {"before": 0.1, "after": 0.027}},
        verification_epochs=3,
        status="REJECT",
        seconds=5,
    )
    rows = store.retrieve("dataset", 2)
    assert rows[0]["verification_epochs"] == 3
    assert rows[0]["base_config_hash"] == "base-a"
    assert store.retrieve("different", 2) == []
