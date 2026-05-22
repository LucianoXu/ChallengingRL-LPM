# Upstream provenance

This directory is an **embedded snapshot** of the official LPM implementation
that accompanies the paper *Beyond Noisy-TVs: Noise-Robust Exploration via
Learning Progress Monitoring* (Hou, An, Du; UC Merced; ICLR 2026).

- **Upstream repository:** https://github.com/Akuna23Matata/LPM_exploration
- **Pinned commit:** `a295e3452491d9475c485a1a66a029eac4d0b55d`
- **Snapshot taken:** 2026-05-22

The upstream `.git/` directory was removed when embedding — these files are
tracked as part of the parent `ChallengingRL` repository, not as a submodule.

To diff against upstream or pull newer changes:

```bash
git clone https://github.com/Akuna23Matata/LPM_exploration.git /tmp/lpm_upstream
diff -ruN /tmp/lpm_upstream LPM_exploration
```

## Local additions / deviations

Any files we modify or add inside `LPM_exploration/` (e.g. fixes, configs,
new ablation scripts) should be noted here so reviewers can tell what is
ours vs. what is upstream.

- *(none yet)*
