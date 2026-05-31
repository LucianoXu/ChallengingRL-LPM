# from .lbs import LBS  # local fix: lbs.py absent in snapshot, unused by LPM path
from .improve import LearningProgressCuriosity
from .ama import AMAPix2PixCuriosity
from .unet_improve import UNetLearningProgressCuriosity
from .RND import RandomNetworkDistillationCuriosity
from .icm import IntrinsicCuriosityModuleCuriosity
from .ensemble import EnsembleDisagreementCuriosity
# from .tdd import TDDNetwork, TemporalDistanceDensityCuriosity  # local fix: tdd.py absent; main.py uses tdd2 directly
# from .tdd2 import TDDNetwork2, TemporalDistanceDensityCuriosity2

print("=== NEW CODE VERSION ===")  # Add this as first line in __init__