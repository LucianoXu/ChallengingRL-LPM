"""Tests for the trajectory-GIF rendering helpers.

Pure rendering helpers (no env, no model) plus the egocentric capture wrapper.
End-to-end rendering against a trained model is exercised by the `--smoke` CLI.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, TILE_PIXELS  # noqa: E402

import gif_gallery as gg  # noqa: E402
from wrappers.ego_capture import EgoCaptureWrapper  # noqa: E402


def _valid_ego_image():
    """A decodable 7x7x3 obs: all empty, one green goal at cell (x=1, y=2)."""
    img = np.zeros((7, 7, 3), dtype=np.uint8)
    img[:, :, 0] = OBJECT_TO_IDX["empty"]
    img[1, 2] = [OBJECT_TO_IDX["goal"], COLOR_TO_IDX["green"], 0]
    return img


def test_render_ego_shape_and_sprite():
    rgb = gg.render_ego(_valid_ego_image())
    assert rgb.shape == (7 * TILE_PIXELS, 7 * TILE_PIXELS, 3)
    # the goal cell (x=1,y=2) -> tile region [y, x] should read green-ish
    t = TILE_PIXELS
    tile = rgb[2 * t:3 * t, 1 * t:2 * t].reshape(-1, 3).mean(0)
    assert tile[1] > tile[0] and tile[1] > tile[2]


def test_render_ego_handles_out_of_range_noise_codes():
    """Grid.decode raises on raw noise codes; render_ego must sanitize and
    still produce a sprite for the hallucinated object (no exception)."""
    noisy = _valid_ego_image()
    noisy[4, 5] = [6, 8, 9]      # object=ball(valid), color 8 + state 9 out-of-range
    noisy[0, 0] = [9, 7, 2]      # object=lava, color 7 out-of-range
    rgb = gg.render_ego(noisy)   # must not raise
    assert rgb.shape == (7 * TILE_PIXELS, 7 * TILE_PIXELS, 3)


def test_object_change_mask():
    true = _valid_ego_image()
    assert not gg.object_change_mask(true, true).any()     # identical -> none
    assert not gg.object_change_mask(true, None).any()     # clean -> none
    noisy = true.copy()
    noisy[4, 5, 0] = OBJECT_TO_IDX["key"]                   # object changed
    noisy[3, 3, 1] = (noisy[3, 3, 1] + 1) % 6              # only color changed
    mask = gg.object_change_mask(true, noisy)
    assert mask[4, 5] and not mask[3, 3] and mask.sum() == 1


def test_mark_corruption_localized():
    ego = gg.render_ego(_valid_ego_image())
    mask = np.zeros((7, 7), dtype=bool)
    mask[4, 5] = True
    out = gg.mark_corruption(ego, mask)
    assert out.shape == ego.shape
    t = TILE_PIXELS
    # marked tile changed (magenta outline)...
    assert not np.array_equal(out[5 * t:6 * t, 4 * t:5 * t],
                              ego[5 * t:6 * t, 4 * t:5 * t])
    # ...an unmarked tile is byte-identical.
    assert np.array_equal(out[2 * t:3 * t, 1 * t:2 * t],
                          ego[2 * t:3 * t, 1 * t:2 * t])


def test_mark_corruption_empty_is_noop():
    ego = gg.render_ego(_valid_ego_image())
    out = gg.mark_corruption(ego, np.zeros((7, 7), dtype=bool))
    assert np.array_equal(out, ego)


def test_draw_trail_shape_and_current_marker():
    base = np.zeros((5 * TILE_PIXELS, 5 * TILE_PIXELS, 3), dtype=np.uint8)
    out = gg.draw_trail(base, [(0, 0), (1, 0), (2, 1)])
    assert out.shape == base.shape
    # current cell (x=2, y=1) center should be marked (cyan -> non-zero G & B)
    t = TILE_PIXELS
    cx, cy = int(2.5 * t), int(1.5 * t)
    px = out[cy, cx]
    assert px[1] > 100 and px[2] > 100


def test_compose_panels_shape():
    left = np.zeros((160, 160, 3), dtype=np.uint8)
    right = np.zeros((224, 224, 3), dtype=np.uint8)
    frame = gg.compose_panels(left, right, "title")
    assert frame.shape[0] == gg.PANEL_H + gg.HEADER_H
    # body width = scaled-left + divider + scaled-right
    lw = round(160 * gg.PANEL_H / 160)
    rw = round(224 * gg.PANEL_H / 224)
    assert frame.shape[1] == lw + gg.DIVIDER + rw
    assert frame.shape[2] == 3


def test_pad_freeze():
    frames = [np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)]
    padded = gg.pad_freeze(frames, 5)
    assert len(padded) == 5
    assert np.array_equal(padded[-1], frames[-1])          # last frame repeated
    assert gg.pad_freeze([], 3) == []


def test_contact_strip_shape():
    def block(n, w):
        return [np.full((40, w, 3), 30, np.uint8) for _ in range(n)]
    stages = {"untrained": block(2, 50), "mid": block(4, 50), "final": block(3, 50)}
    strip = gg.contact_strip(stages)
    assert len(strip) == 4                                  # padded to max length
    assert strip[0].shape[1] == 50 * 3 + gg.DIVIDER * 2     # 3 cols + 2 dividers
    assert strip[0].shape[0] == 40


def test_ego_capture_wrapper_passthrough():
    import gymnasium as gym
    import minigrid  # noqa: F401  (registers the envs)
    base = gym.make("MiniGrid-DoorKey-5x5-v0")
    cap = EgoCaptureWrapper(base)
    obs, _ = cap.reset(seed=1)
    assert cap.last_image is not None
    assert np.array_equal(cap.last_image, obs["image"])     # records the image
    assert "image" in obs and "direction" in obs            # obs passed through
    cap.close()
