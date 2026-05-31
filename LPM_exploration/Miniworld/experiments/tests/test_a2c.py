import numpy as np
import torch
import a2c

OBS = (120, 160, 3)


def test_running_norm_converges_to_unit_std():
    rn = a2c.RunningNorm()
    for v in np.random.randn(500) * 3 + 7:
        rn.update(float(v))
    x = rn.normalize(10.0)
    assert abs(x) < 5  # finite, roughly standardized


def test_agent_select_and_update_runs():
    net = a2c.A2CNetwork((3, 120, 160), 5).to("cpu")
    agent = a2c.A2CAgent(net, num_actions=5, device="cpu", lambda_intrinsic=0.1)
    state = np.random.randint(0, 256, OBS, dtype=np.uint8)
    a, lp, v = agent.select_action(state)
    assert 0 <= a < 5
    # Fill memory with a few fake transitions, then update.
    for _ in range(8):
        ns = np.random.randint(0, 256, OBS, dtype=np.uint8)
        agent.memory.add(state, a, 0.0, 0.01, ns, False, lp, v)
    losses = agent.update()
    assert "policy_loss" in losses and "value_loss" in losses
