"""Keyboard-controlled play tool for the LPM Miniworld scenarios.

Re-export the env registration so ``from miniworld_play import envs`` is enough
to trigger ``register()`` calls.
"""
from miniworld_play import envs as envs  # noqa: F401
