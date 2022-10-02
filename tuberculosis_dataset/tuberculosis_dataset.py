import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import pandas as pd
from os import path


class TuberculosisDataset(Dataset):
    """TuberculosisDataset class is used to create a dataset object for the tuberculosis dataset."""

    def __init__(
        self,
        dataframe,
        transforms=None,
        root_dir=None,
        path="path",
        abnormal="abnormal",
        tuberculosis="tuberculosis",
        labels_source="all",
        mode="RGB",
    ):
        self._dataframe = (
            pd.read_csv(dataframe) if isinstance(dataframe, str) else dataframe
        )
        self.transforms = transforms
        self.root_dir = root_dir if root_dir else "./"
        self.path = path
        self.abnormal = abnormal
        self.tuberculosis = tuberculosis
        self.labels_source = (
            [self.abnormal, self.tuberculosis]
            if labels_source == "all"
            else [labels_source]
        )
        assert self.labels_source in [
            [self.abnormal],
            [self.tuberculosis],
            [self.abnormal, self.tuberculosis],
        ], "Invalid Labels Source"

        assert mode.upper() in ["RGB", "GRAY"], "Invalid Image Read Mode"
        self.mode = ImageReadMode.RGB if mode.upper() == "RGB" else ImageReadMode.GRAY

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = path.join(self.root_dir, str(self._dataframe.iloc[index][self.path]))
        image = to_pil_image(read_image(img_path, self.mode))
        # image = Image.open(img_path)

        if self.transforms:
            image = self.transforms(image)

        if len(self.labels_source) == 1:
            labels = self._dataframe.iloc[index][self.labels_source[0]].astype("float")
        if len(self.labels_source) == 2:
            labels = self._dataframe.iloc[index][self.labels_source].astype("float")

        return image, torch.tensor(labels).type(torch.FloatTensor)
