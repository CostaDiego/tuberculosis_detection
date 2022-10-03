__all__ = ["classifier", "criterion", "utils"]

from .classifier import VGG16DoubleHeadClassifier, VGG16SingleHeadClassifier
from .criterion import get_2heads_criterion
from .utils import train_model
