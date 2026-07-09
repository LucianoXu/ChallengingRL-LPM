"""Method-string parsing shared across the MiniGrid pipeline.

A method string carries both the intrinsic-reward kind and the policy
architecture: a trailing '_lstm' selects a RecurrentPPO LSTM policy, while the
base ('rnd' / 'lpm' / 'icm') selects the intrinsic-reward wrapper.
"""

# Base methods that add an intrinsic reward in the training path. (count is
# implemented in env_factory but dormant — kept out so existing behavior, where
# train_one only treats these bases as intrinsic, is preserved.)
INTRINSIC_BASES = ("rnd", "lpm", "icm")

_LSTM_SUFFIX = "_lstm"


def base_intrinsic(method: str) -> str:
    """Strip a trailing '_lstm' policy-architecture suffix."""
    if method.endswith(_LSTM_SUFFIX):
        return method[: -len(_LSTM_SUFFIX)]
    return method


def is_recurrent(method: str) -> bool:
    """True iff the method uses a recurrent (LSTM) policy."""
    return method.endswith(_LSTM_SUFFIX)


def is_intrinsic(method: str) -> bool:
    """True iff the method adds an intrinsic reward (ignoring policy arch)."""
    return base_intrinsic(method) in INTRINSIC_BASES
