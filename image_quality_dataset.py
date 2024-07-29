from typing import Dict, List

import numpy as np
from PIL import Image
from skimage.io import imread


class ImageQualityDataset(Dataset):
    # Require that the image_map argument is a list with dictionary values
    def __init__(self, image_map: List[Dict]):
        "initialize the directory containing images"
        self.image_map = image_map

    def read_image(self, image_path):
        "returns information from the 3 channels for every pixel"
        image = imread(image_path)
        return image

    def __len__(self):
        "returns the number of samples"
        return len(self.image_map)

    def __getitem__(self, idx):
        "making sure image_map has 'path' to frame and 'target' that is 0 or 1"
        sample = self.image_map[idx]
        assert "path" in sample, "Each sample must have a path property"
        assert "target" in sample, "Each sample must have a target property"
        image_path = sample["path"]
        image = self.read_image(image_path)
        target = sample["target"]
        assert isinstance(target, int), "Your target must be an integer"
        assert target in [0, 1], "Your target must be 0 or 1"

        image = Image.fromarray(image)
        image = image.resize((224, 224))
        image = np.array(image)

        # Change from bytes [0,255] to a float [0,1]
        image = image / 255
        image = image.astype(np.float32)

        return image, target