"""Config matrix for the MiniGrid trajectory-GIF gallery.

Single source of truth shared by `gif_gallery.py` (rendering) and
`make_stage_snapshots.py` (optional clean re-training). The full matrix is
3 envs x {clean, noisy} x {none, entropy, rnd, icm, lpm} = 30 configs, each rendered
at three stages (untrained / mid / final) sourced from existing on-disk
checkpoints. Cells without a final model on disk are skipped automatically.

See docs/superpowers/specs/2026-06-23-minigrid-trajectory-gifs-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import MODELS_DIR, EXPR_DATA

# Untrained (random) snapshots live here; trained stages come from the study's
# own models/ dir (final .zip + per-chunk best-eval checkpoints under best/).
SNAPSHOTS_DIR = MODELS_DIR.parent / "ppo_gif_snapshots"
GIFS_DIR = EXPR_DATA / "figures" / "gifs"

NOISE_PROB = 0.10  # study default for the noise variants

# short slug -> (env id, difficulty tier)
ENVS = {
    "doorkey-5x5":  ("MiniGrid-DoorKey-5x5-v0",  "easy"),
    "fourrooms":    ("MiniGrid-FourRooms-v0",    "medium"),
    "multiroom-n6": ("MiniGrid-MultiRoom-N6-v0", "hard"),
}
METHODS = ["none", "entropy", "rnd", "icm", "lpm"]


@dataclass(frozen=True)
class GifConfig:
    slug: str
    env_id: str
    noise: bool
    method: str
    tier: str
    seed: int = 1               # training seed (which on-disk run to read)
    render_seed: int | None = None   # fixed layout; None -> auto-pick at render

    @property
    def intrinsic(self) -> bool:
        return self.method in ("rnd", "icm", "lpm")

    @property
    def variant(self) -> str:
        base = "intrinsic" if self.intrinsic else "baseline"
        return f"{base}_{'noise' if self.noise else 'no_noise'}"

    @property
    def run_name(self) -> str:
        """Base run name (study scheme)."""
        tag = f"__np{NOISE_PROB:g}" if self.noise else ""
        name = f"{self.env_id}__{self.variant}__{self.method}__seed_{self.seed}{tag}"
        return name.replace("/", "_")

    @property
    def finding(self) -> str:
        return f"{self.tier} env · {self.method} · {'noisy obs' if self.noise else 'clean'}"

    # --- Stage model sources (reuse existing on-disk study checkpoints) -------
    def untrained_path(self):
        return SNAPSHOTS_DIR / f"{self.run_name}__step0.zip"

    def final_path(self):
        return MODELS_DIR / f"{self.run_name}.zip"

    def final_steps(self) -> int:
        p = MODELS_DIR / f"{self.run_name}.progress"
        try:
            return int(p.read_text().strip())
        except (OSError, ValueError):
            return 0

    def mid_path(self):
        """(path, chunk_start_step) of the best-eval checkpoint nearest half the
        final step budget, or None if no per-chunk checkpoints exist. c0 (best of
        the first chunk, already competent) is excluded so mid sits between the
        random untrained stage and the final stage."""
        base = MODELS_DIR / "best" / self.run_name
        chunks = sorted(int(p.name[1:]) for p in base.glob("c*")
                        if p.is_dir() and p.name[1:].isdigit())
        chunks = [c for c in chunks if c > 0] or chunks
        if not chunks:
            return None
        target = max(self.final_steps() // 2, 1)
        c = min(chunks, key=lambda x: abs(x - target))
        return base / f"c{c}" / "best_model.zip", c

    def stage_sources(self):
        """Ordered [(stage, model_path, step_label, deterministic)].
        Untrained samples actions (a fresh argmax policy just spins); the trained
        stages are deterministic."""
        out = [("untrained", self.untrained_path(), 0, False)]
        mid = self.mid_path()
        if mid is not None:
            path, step = mid
            out.append(("mid", path, step, True))
        out.append(("final", self.final_path(), self.final_steps(), True))
        return out


def _build_matrix() -> list[GifConfig]:
    cfgs = []
    for eshort, (env_id, tier) in ENVS.items():
        for noise in (False, True):
            for method in METHODS:
                slug = f"{eshort}_{'noisy' if noise else 'clean'}_{method}"
                c = GifConfig(slug, env_id, noise, method, tier)
                if c.final_path().exists():      # only renderable cells
                    cfgs.append(c)
    return cfgs


GIF_CONFIGS: list[GifConfig] = _build_matrix()


def get_config(slug: str) -> GifConfig:
    for c in GIF_CONFIGS:
        if c.slug == slug:
            return c
    raise KeyError(f"unknown gif config slug: {slug!r}; "
                   f"known: {[c.slug for c in GIF_CONFIGS]}")
