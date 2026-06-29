from .registry import LOSSES
from .losses import (  # noqa: F401 – triggers @LOSSES.register decorators
    BCELossWrapper,
    CombinedLoss,
    CrossEntropyLossWrapper,
    DiceLoss,
    FocalLoss,
    HuberLossWrapper,
    L1LossWrapper,
    MSELossWrapper,
)
