"""Miniworld environments from Hou et al. 2026 (LPM), ported for human play.

Three variants, all sharing the same 4-room geometry hand-designed in the
upstream notebooks at LPM_exploration/Miniworld/:

  - NoNoiseEnv         : clean baseline.
  - ActionNoiseEnv     : action 4 returns a random CIFAR-10 image (the agent
                         can "stare at the noisy TV"); the grass-textured
                         noise wall is also replaced with random RGB.
  - NoisyTVEnv         : grass-textured noise wall is replaced with random
                         RGB every step; 25% sticky-action stochasticity.

All variants register with gymnasium so the play tool can resolve them by id.

The geometry, the green-pixel detection mask (including its uint8 quirk),
the step parameters (fwd_step=0.4, turn_step=5°), the sticky-action
probability (0.25), the agent start ([2, 0, 1], facing -π/2), and the
visited-state grid (72 × 48 cells) are reproduced verbatim from the
upstream notebooks so a human plays exactly what the agent trains against.
"""

from __future__ import annotations

import math
import os
import random
from typing import Any, Optional

import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.registration import register

from miniworld.miniworld import MiniWorldEnv


# Default to the upstream CIFAR cache path so we don't redownload it.
_DEFAULT_CIFAR_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "LPM_exploration",
    "Miniworld",
    "data",
)


def _replace_green_with_random(observation: np.ndarray) -> np.ndarray:
    """Per-pixel green→random RGB. Bit-for-bit identical to upstream.

    The R<800 clause is an upstream quirk (always true for uint8) — kept so
    behaviour matches the paper. The substantive condition is G>80 ∧ B<50.
    """
    modified_obs = observation.copy()
    green_mask = (
        (observation[:, :, 0] < 800)  # vestigial; uint8 max is 255
        & (observation[:, :, 1] > 80)
        & (observation[:, :, 2] < 50)
    )
    n_green = int(green_mask.sum())
    if n_green == 0:
        return modified_obs

    # Vectorised replacement (upstream uses a python for-loop; result identical).
    rand_rgb = np.random.randint(0, 256, size=(n_green, 3), dtype=np.uint8)
    modified_obs[green_mask] = rand_rgb
    return modified_obs


class _BaseMazeEnv(MiniWorldEnv, utils.EzPickle):
    """Geometry + action handling common to all three variants.

    Variants override :meth:`_apply_obs_transform` (no-op by default) and
    :meth:`_maybe_sticky` (no-op by default).
    """

    FWD_STEP = 0.4
    TURN_STEP_DEG = 5
    OBS_WIDTH = 160
    OBS_HEIGHT = 120

    def __init__(
        self,
        max_episode_steps: int = 50000,
        obs_width: int = OBS_WIDTH,
        obs_height: int = OBS_HEIGHT,
        render_mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.visited_states = np.zeros((18 * 4, 12 * 4), dtype=np.uint8)
        self.prev_action: Optional[int] = None

        MiniWorldEnv.__init__(
            self,
            max_episode_steps=max_episode_steps,
            obs_width=obs_width,
            obs_height=obs_height,
            render_mode=render_mode,
            **kwargs,
        )
        utils.EzPickle.__init__(self)
        # 5 discrete actions: 0=turn_left, 1=turn_right, 2=move_forward,
        # 3=move_back, 4=variant-specific extra action (see subclasses).
        self.action_space = spaces.Discrete(5)

    # ----- geometry (identical across all three notebooks) -----
    def _gen_world(self) -> None:
        room1 = self.add_rect_room(
            min_x=0, max_x=4, min_z=0, max_z=8,
            wall_tex="cardboard", floor_tex="stucco",
        )
        room2 = self.add_rect_room(
            min_x=14, max_x=18, min_z=0, max_z=8,
            wall_tex="asphalt", floor_tex="stucco",
        )
        # Thin strip with grass texture — this is the "noise wall".
        room3 = self.add_rect_room(
            min_x=0, max_x=18, min_z=8, max_z=8.1,
            wall_tex="grass", floor_tex="stucco",
        )
        room4 = self.add_rect_room(
            min_x=0, max_x=18, min_z=8.1, max_z=12,
            floor_tex="stucco",
        )

        self.connect_rooms(room1, room3, min_x=0, max_x=4)
        self.connect_rooms(room2, room3, min_x=14, max_x=18)
        self.connect_rooms(room3, room4, min_x=0, max_x=18)

        self.place_agent(pos=[2, 0, 1], dir=-math.pi / 2, room=room1)

    # ----- per-variant hooks -----
    def _apply_obs_transform(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def _maybe_sticky(self, action: int) -> tuple[int, bool]:
        return action, False

    def _handle_special_action(self, action: int) -> Optional[np.ndarray]:
        """Return an observation if `action` is a variant-specific extra
        action (e.g. action 4 = "look at noisy TV" in ActionNoiseEnv);
        return None to fall through to the normal move/turn handling.
        """
        return None

    # ----- step (action semantics shared) -----
    def step(self, action):  # type: ignore[override]
        action = int(action)
        action, sticky_replayed = self._maybe_sticky(action)
        self.prev_action = action
        self.step_count += 1

        special_obs = self._handle_special_action(action)
        if special_obs is not None:
            obs = special_obs
            agent_x = float(self.agent.pos[0])
            agent_z = float(self.agent.pos[2])
        else:
            if action == self.actions.move_forward:
                self.move_agent(self.FWD_STEP, 0)
            elif action == self.actions.move_back:
                self.move_agent(-self.FWD_STEP, 0)
            elif action == self.actions.turn_left:
                self.turn_agent(self.TURN_STEP_DEG)
            elif action == self.actions.turn_right:
                self.turn_agent(-self.TURN_STEP_DEG)
            # else: any other id (including 4 in non-action-noise variants) is
            # a no-op, matching upstream.

            obs = self.render_obs()
            agent_x = float(self.agent.pos[0])
            agent_z = float(self.agent.pos[2])
            ix = max(0, min(self.visited_states.shape[0] - 1, int(agent_x * 4)))
            iz = max(0, min(self.visited_states.shape[1] - 1, int(agent_z * 2 * 2)))  # = agent_z*4
            self.visited_states[ix, iz] = 1
            obs = self._apply_obs_transform(obs)

        reward = 0.0
        terminated = False
        truncated = self.step_count >= self.max_episode_steps
        info = {
            "visited_state": int(self.visited_states.sum()),
            "pos": [agent_x, agent_z],
            "dir": float(self.agent.dir),
            "sticky_replayed": sticky_replayed,
            "action_id": int(action),
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        self.visited_states.fill(0)
        self.prev_action = None
        return super().reset(seed=seed, options=options)


class NoNoiseEnv(_BaseMazeEnv):
    """Clean baseline: vanilla geometry, no noise, no sticky actions."""


class NoisyTVEnv(_BaseMazeEnv):
    """The headline variant: per-step random RGB replacement on the grass
    wall + 25% sticky-action stochasticity.

    Sticky probability can be overridden at construction (set
    ``sticky_prob=0.0`` to disable while keeping the visual noise).
    """

    def __init__(self, sticky_prob: float = 0.25, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.sticky_prob = float(sticky_prob)

    def _apply_obs_transform(self, obs: np.ndarray) -> np.ndarray:
        return _replace_green_with_random(obs)

    def _maybe_sticky(self, action: int) -> tuple[int, bool]:
        if (
            self.prev_action is not None
            and self.sticky_prob > 0.0
            and random.random() < self.sticky_prob
        ):
            return self.prev_action, True
        return action, False


class ActionNoiseEnv(_BaseMazeEnv):
    """Action-conditioned noise: action 4 = "look at the noisy TV", returns
    a stretched random CIFAR-10 image as the observation, skipping the env
    step. Normal moves still apply the grass-wall noise transform.

    Sticky actions are also present in upstream — kept here for parity.
    """

    def __init__(
        self,
        sticky_prob: float = 0.25,
        cifar_root: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sticky_prob = float(sticky_prob)
        self._cifar_root = cifar_root or _DEFAULT_CIFAR_ROOT
        self._cifar_dataset = None  # lazy

    def _ensure_cifar(self):
        if self._cifar_dataset is None:
            try:
                import torchvision
                import torchvision.transforms as transforms
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "ActionNoiseEnv needs torchvision for CIFAR-10. "
                    "Install with: uv pip install --python <venv> torchvision"
                ) from e

            transform = transforms.Compose([transforms.ToTensor()])
            os.makedirs(self._cifar_root, exist_ok=True)
            try:
                self._cifar_dataset = torchvision.datasets.CIFAR10(
                    root=self._cifar_root, train=True, download=True, transform=transform,
                )
            except Exception as e:  # network failure → fall back to synthetic
                print(
                    f"[ActionNoiseEnv] CIFAR-10 download failed ({e}); "
                    "falling back to synthetic noise patches."
                )
                self._cifar_dataset = None
        return self._cifar_dataset

    def _random_cifar_obs(self) -> np.ndarray:
        from PIL import Image
        ds = self._ensure_cifar()
        if ds is not None:
            idx = random.randint(0, len(ds) - 1)
            image, _ = ds[idx]
            arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr).resize((self.OBS_WIDTH, self.OBS_HEIGHT), Image.Resampling.BILINEAR)
            return np.array(pil)
        # Fallback: synthetic high-entropy patches
        return np.random.randint(0, 256, size=(self.OBS_HEIGHT, self.OBS_WIDTH, 3), dtype=np.uint8)

    def _handle_special_action(self, action: int) -> Optional[np.ndarray]:
        if action == 4:
            return self._random_cifar_obs()
        return None

    def _apply_obs_transform(self, obs: np.ndarray) -> np.ndarray:
        return _replace_green_with_random(obs)

    def _maybe_sticky(self, action: int) -> tuple[int, bool]:
        if (
            self.prev_action is not None
            and self.sticky_prob > 0.0
            and random.random() < self.sticky_prob
        ):
            return self.prev_action, True
        return action, False


# ----- gymnasium registration -----
register(id="LPMPlay-MazeNoNoise-v0", entry_point=NoNoiseEnv, max_episode_steps=50000)
register(id="LPMPlay-MazeNoisyTV-v0", entry_point=NoisyTVEnv, max_episode_steps=50000)
register(id="LPMPlay-MazeActionNoise-v0", entry_point=ActionNoiseEnv, max_episode_steps=50000)


VARIANT_TO_ID = {
    "nonoise": "LPMPlay-MazeNoNoise-v0",
    "noisy_tv": "LPMPlay-MazeNoisyTV-v0",
    "action_noise": "LPMPlay-MazeActionNoise-v0",
}
