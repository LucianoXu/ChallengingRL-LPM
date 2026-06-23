import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import analyze


def test_parse_run_name():
    p = analyze.parse_run_name(
        "MiniGrid-FourRooms-v0__intrinsic_noise__lpm__seed_3__beta0.05")
    assert p == {"env": "MiniGrid-FourRooms-v0", "variant": "intrinsic_noise",
                 "method": "lpm", "seed": 3, "beta": "0.05", "np": None}


def test_parse_run_name_noise_tag():
    p = analyze.parse_run_name(
        "MiniGrid-FourRooms-v0__intrinsic_noise__rnd__seed_1__np0.2")
    assert p["np"] == "0.2" and p["beta"] is None and p["method"] == "rnd"


def test_parse_run_name_no_beta():
    p = analyze.parse_run_name(
        "MiniGrid-Empty-8x8-v0__baseline_no_noise__none__seed_1")
    assert p["beta"] is None and p["method"] == "none" and p["seed"] == 1


def test_parse_run_name_recurrent_method():
    p = analyze.parse_run_name(
        "MiniGrid-DoorKey-5x5-v0__intrinsic_noise__rnd_lstm__seed_2__np0.1")
    assert p["method"] == "rnd_lstm" and p["np"] == "0.1" and p["seed"] == 2


def test_parse_run_name_recurrent_clean():
    p = analyze.parse_run_name(
        "MiniGrid-FourRooms-v0__intrinsic_no_noise__lpm_lstm__seed_1")
    assert p["method"] == "lpm_lstm" and p["np"] is None


def test_load_eval_npz(tmp_path):
    d = tmp_path / "eval" / "MiniGrid-Empty-8x8-v0__baseline_no_noise__none__seed_1"
    d.mkdir(parents=True)
    np.savez(d / "evaluations.npz",
             timesteps=np.array([100, 200]),
             results=np.array([[0.1, 0.3], [0.5, 0.7]]),
             ep_lengths=np.array([[10, 10], [9, 9]]))
    rows = analyze.load_eval_npz(str(d / "evaluations.npz"))
    assert [r["timestep"] for r in rows] == [100, 200]
    assert abs(rows[1]["mean_return"] - 0.6) < 1e-9
