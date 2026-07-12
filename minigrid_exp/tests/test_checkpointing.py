import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train


class _FakeModel:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save(self, path):
        Path(path).write_bytes(self.payload)


class _BaselineEnv:
    pass


def test_two_slot_checkpoint_commit_and_rotation(tmp_path):
    run_name = "test_run"
    model = _FakeModel(b"first")
    env = _BaselineEnv()

    slot = train._save_resume_checkpoint(
        model, env, run_name, tmp_path, completed_steps=100,
        active_slot=None, expect_intrinsic=False,
    )
    assert slot == 0
    assert train._read_progress(run_name, model_dir=tmp_path) == 100
    assert train._active_resume_slot(run_name, tmp_path, 100) == 0
    assert (tmp_path / f"{run_name}.zip").read_bytes() == b"first"

    model.payload = b"second"
    slot = train._save_resume_checkpoint(
        model, env, run_name, tmp_path, completed_steps=200,
        active_slot=slot, expect_intrinsic=False,
    )
    assert slot == 1
    assert train._read_progress(run_name, model_dir=tmp_path) == 200
    assert train._active_resume_slot(run_name, tmp_path, 200) == 1
    assert (tmp_path / f"{run_name}.zip").read_bytes() == b"second"


def test_failed_state_save_does_not_advance_progress(tmp_path, monkeypatch):
    run_name = "test_run"
    model = _FakeModel(b"committed")
    env = _BaselineEnv()
    slot = train._save_resume_checkpoint(
        model, env, run_name, tmp_path, completed_steps=100,
        active_slot=None, expect_intrinsic=False,
    )

    def fail_save(*args, **kwargs):
        raise OSError("simulated checkpoint failure")

    monkeypatch.setattr(train.th, "save", fail_save)
    model.payload = b"uncommitted"
    with pytest.raises(OSError, match="simulated checkpoint failure"):
        train._save_resume_checkpoint(
            model, env, run_name, tmp_path, completed_steps=200,
            active_slot=slot, expect_intrinsic=False,
        )

    assert train._read_progress(run_name, model_dir=tmp_path) == 100
    assert train._active_resume_slot(run_name, tmp_path, 100) == 0
    assert (tmp_path / f"{run_name}.zip").read_bytes() == b"committed"


def test_intrinsic_checkpoint_requires_shared_wrapper(tmp_path):
    with pytest.raises(RuntimeError, match="no shared intrinsic wrapper"):
        train._save_resume_checkpoint(
            _FakeModel(b"model"), _BaselineEnv(), "bad", tmp_path,
            completed_steps=100, active_slot=None, expect_intrinsic=True,
        )


def test_train_agent_restores_intrinsic_state_across_chunks(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    model_dir = tmp_path / "models"
    log_dir.mkdir()
    model_dir.mkdir()
    monkeypatch.setattr(train, "PPO_VEC_ENV", "dummy")
    monkeypatch.setattr(
        train,
        "PPO_HYPERPARAMS",
        {**train.PPO_HYPERPARAMS, "n_steps": 8, "batch_size": 8, "n_epochs": 1},
    )
    kwargs = dict(
        env_id="MiniGrid-Empty-8x8-v0",
        variant_name="intrinsic_no_noise",
        intrinsic=True,
        noise=False,
        seed=1,
        log_dir=log_dir,
        model_dir=model_dir,
        method="rnd",
        chunk_steps=64,
    )
    run_name = "MiniGrid-Empty-8x8-v0__intrinsic_no_noise__rnd__seed_1"

    train.train_agent(total_timesteps=64, **kwargs)
    first_slot = train._active_resume_slot(run_name, model_dir, 64)
    first_state = train._load_torch_checkpoint(
        train._resume_path(model_dir, run_name, first_slot, "state.pt"), "cpu"
    )
    first_obs_count = first_state["intrinsic"]["model"]["obs_rms"]["count"]

    train.train_agent(total_timesteps=128, **kwargs)
    second_slot = train._active_resume_slot(run_name, model_dir, 128)
    second_state = train._load_torch_checkpoint(
        train._resume_path(model_dir, run_name, second_slot, "state.pt"), "cpu"
    )
    second_obs_count = second_state["intrinsic"]["model"]["obs_rms"]["count"]

    assert first_slot == 0 and second_slot == 1
    assert train._read_progress(run_name, model_dir=model_dir) == 128
    assert second_obs_count > first_obs_count
    assert (model_dir / f"{run_name}.zip").exists()
    assert (log_dir / f"{run_name}__c0.monitor.csv").exists()
    assert (log_dir / f"{run_name}__c64.monitor.csv").exists()
