import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv

import pytest
from stable_baselines3.common.monitor import Monitor

from wrappers.env_factory import make_env


def _rollout_until_done(env, max_steps=5000):
    """Step with random actions until the first episode terminates or truncates.
    Returns (per_step_infos, final_info, final_reward)."""
    obs, _ = env.reset()
    per_step_ext = 0.0
    per_step_intr = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        per_step_ext += float(info.get("extrinsic_reward", reward))
        per_step_intr += float(
            info.get("rnd_intrinsic_reward",
                     info.get("lpm_intrinsic_reward",
                              info.get("icm_intrinsic_reward", 0.0)))
        )
        if terminated or truncated:
            return per_step_ext, per_step_intr, info
    raise RuntimeError("No episode ended within max_steps")


def test_lpm_accumulation_matches():
    """ep_extrinsic and ep_intrinsic at episode end must match manual accumulators."""
    env = make_env(
        "MiniGrid-Empty-8x8-v0",
        intrinsic=True,
        noise=False,
        seed=0,
        training=True,
        method="lpm",
        beta=0.05,
    )
    try:
        manual_ext, manual_intr, final_info = _rollout_until_done(env)
    finally:
        env.close()

    assert "ep_extrinsic" in final_info, "ep_extrinsic not in terminal info"
    assert "ep_intrinsic" in final_info, "ep_intrinsic not in terminal info"
    assert abs(final_info["ep_extrinsic"] - manual_ext) < 1e-4, (
        f"ep_extrinsic mismatch: wrapper={final_info['ep_extrinsic']:.6f} "
        f"manual={manual_ext:.6f}"
    )
    assert abs(final_info["ep_intrinsic"] - manual_intr) < 1e-4, (
        f"ep_intrinsic mismatch: wrapper={final_info['ep_intrinsic']:.6f} "
        f"manual={manual_intr:.6f}"
    )


def test_baseline_zero_intrinsic():
    """Baseline (no intrinsic wrapper): ep_intrinsic must be 0.0 at episode end."""
    env = make_env(
        "MiniGrid-Empty-8x8-v0",
        intrinsic=False,
        noise=False,
        seed=0,
        training=True,
    )
    try:
        _, _, final_info = _rollout_until_done(env)
    finally:
        env.close()

    assert "ep_extrinsic" in final_info, "ep_extrinsic not in terminal info"
    assert "ep_intrinsic" in final_info, "ep_intrinsic not in terminal info"
    assert final_info["ep_intrinsic"] == 0.0, (
        f"Expected ep_intrinsic == 0.0 for baseline, got {final_info['ep_intrinsic']}"
    )


def test_monitor_persists_columns(tmp_path):
    """Monitor with info_keywords writes ep_extrinsic and ep_intrinsic as CSV columns."""
    env = make_env(
        "MiniGrid-Empty-8x8-v0",
        intrinsic=True,
        noise=False,
        seed=1,
        training=True,
        method="lpm",
        beta=0.05,
    )
    monitor_filename = str(tmp_path / "test_run")
    mon = Monitor(
        env,
        filename=monitor_filename,
        info_keywords=("ep_extrinsic", "ep_intrinsic"),
    )
    # Roll until at least one episode completes
    obs, _ = mon.reset()
    done_count = 0
    for _ in range(5000):
        action = mon.action_space.sample()
        obs, reward, terminated, truncated, info = mon.step(action)
        if terminated or truncated:
            obs, _ = mon.reset()
            done_count += 1
            if done_count >= 1:
                break
    mon.close()

    # Find the CSV file (Monitor appends .monitor.csv)
    csv_files = list(tmp_path.glob("*.monitor.csv"))
    assert csv_files, f"No monitor CSV found in {tmp_path}"
    csv_path = csv_files[0]

    with open(csv_path, newline="") as f:
        # First line is a comment starting with '#', second line is the header
        lines = f.readlines()

    header_line = None
    for line in lines:
        if not line.startswith("#"):
            header_line = line.strip()
            break

    assert header_line is not None, "No header line found in monitor CSV"
    columns = [c.strip() for c in header_line.split(",")]
    assert "ep_extrinsic" in columns, (
        f"ep_extrinsic not in CSV header. Header: {header_line}"
    )
    assert "ep_intrinsic" in columns, (
        f"ep_intrinsic not in CSV header. Header: {header_line}"
    )

    # Report the header for the caller
    print(f"\nmonitor.csv header: {header_line}")
