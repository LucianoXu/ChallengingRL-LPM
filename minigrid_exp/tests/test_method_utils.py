import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from method_utils import base_intrinsic, is_recurrent, is_intrinsic


def test_base_intrinsic_strips_lstm():
    assert base_intrinsic("rnd_lstm") == "rnd"
    assert base_intrinsic("lpm_lstm") == "lpm"
    assert base_intrinsic("icm_lstm") == "icm"
    assert base_intrinsic("rnd") == "rnd"
    assert base_intrinsic("none") == "none"


def test_is_recurrent():
    assert is_recurrent("rnd_lstm") and is_recurrent("lpm_lstm") and is_recurrent("icm_lstm")
    assert not is_recurrent("rnd") and not is_recurrent("none")


def test_is_intrinsic():
    assert is_intrinsic("rnd") and is_intrinsic("lpm") and is_intrinsic("icm")
    assert is_intrinsic("rnd_lstm") and is_intrinsic("lpm_lstm") and is_intrinsic("icm_lstm")
    assert not is_intrinsic("none") and not is_intrinsic("entropy")
