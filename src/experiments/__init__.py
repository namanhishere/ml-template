from .registry import EXPERIMENTS
from .classification import ClassificationExperiment  # noqa: F401 — triggers @EXPERIMENTS.register
from .segmentation import SegmentationExperiment  # noqa: F401
from .regression import RegressionExperiment  # noqa: F401
