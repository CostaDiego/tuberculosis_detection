__all__ = ["classifier"]

from .classifier import VGG16DoubleHeadClassifier, VGG16SingleHeadClassifier
from .criterion import get_2heads_criterion
