"""
Script for saving model predictions to a csv file. Each row contains the path to the image and the model's class prediction and associated probability.

Author: Helene Li
"""

# Importing necessary libraries
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

project = Path("/home/brennamacaulay/Downloads")

images = list(project.glob("20230627/*/*.png"))

imagemap = [{"path": p, "target": 0} for p in images]

dataset = ImageQualityDataset(imagemap)

preds = []

for i in tqdm(range(len(dataset))):
    x, _ = dataset[i]
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
            "path": images[i],
            "pred": pred.argmax().item(),
            "score": predicted_prob.item(),
        }
    )

df = pd.DataFrame(preds)
df.to_csv("20230627Quality.csv", index=False)
