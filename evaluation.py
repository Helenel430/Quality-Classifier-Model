"""
Script for saving model predictions to a csv file. Each row contains the path to the image and the model's class prediction and associated probability.

Author: Helene Li
"""

# Importing necessary libraries
from pathlib import Path

import torch
from tqdm.auto import tqdm

project = Path("/home/brennamacaulay/Downloads")

sample_images = list(project.glob("20230627/*/*.png"))

sample_imagemap = [{"path": p, "target": 0} for p in sample_images]

sample_dataset = ImageQualityDataset(sample_imagemap)

preds = []

for i in tqdm(range(len(sample_dataset))):
    x, _ = sample_dataset[i]
    x = torch.from_numpy(x).to(device)

    # Add extra dimension at the start for the batch
    x = x.unsqueeze(dim=0)
    x = x.permute(0, 3, 1, 2)
    pred = model(x)

    # Getting predicted classes and probability of prediction
    probabilities = torch.softmax(pred, dim=1)
    predicted_prob = probabilities.max(dim=1)[0]

    preds.append(
        {
            "path": sample_images[i],
            "pred": pred.argmax().item(),
            "score": predicted_prob.item(),
        }
    )
