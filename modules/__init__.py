__all__ = ["classifier"]

from .classifier import VGG16DoubleHeadClassifier, VGG16SingleHeadClassifier
from .utils import get_2heads_criterion
