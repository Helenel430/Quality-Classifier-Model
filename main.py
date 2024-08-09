# Importing necessary libraries
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

from image_quality_dataset import ImageQualityDataset
from transfer_learning_resnet import TransferLearningResNet

# Defining paths
project = Path("/home/brennamacaulay/Desktop/Stress_Facial_Exp-Helene-2024-03-20")
sampled_frames = list(project.glob("sampled_frames_folder/*/*.png"))

# Making the training (80%), validation (10%), testing datasets (10%)
total_count = len(sampled_frames)
train_count = int(0.7 * total_count)
test_count = int(0.1 * total_count)
# Roughly 20% will be validation data
valid_count = total_count - train_count - test_count

# Ensuring reproducibility
g = torch.Generator().manual_seed(0)

# Splitting the dataset
train_data, valid_data, test_data = random_split(
    sampled_frames, [train_count, valid_count, test_count], generator=g
)
print(train_data)

# Initalize List of Dict
image_map_train = []
image_map_valid = []
image_map_test = []


# Input: data subsets and empty list of Dict
# Output: list of Dict for each subset
def QualityLabel(subset, listOfDict):
    for frame in subset:
        frame_path = Path(frame)
        # Appending the frame path and target values to the dictionary
        if frame_path.parent.stem == "good_frames":
            good_value = 1
            listOfDict.append({"path": frame_path, "target": good_value})
        elif frame_path.parent.stem == "bad_frames":
            bad_value = 0
            listOfDict.append({"path": frame_path, "target": bad_value})
    return listOfDict


image_map_train = QualityLabel(train_data, image_map_train)
image_map_valid = QualityLabel(valid_data, image_map_valid)
image_map_test = QualityLabel(test_data, image_map_test)

# Testing
# Wrapping each list in the dataset
dataset_train = ImageQualityDataset(image_map_train)
dataset_valid = ImageQualityDataset(image_map_valid)
dataset_test = ImageQualityDataset(image_map_test)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Using dataloader to load datasets
batch_size = 8
num_workers = 1

train_dataloader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
    shuffle=True,
)
# Do not shuffle validation and testing for evaluation consistency
valid_dataloader = DataLoader(
    dataset_valid,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
    shuffle=False,
)
test_dataloader = DataLoader(
    dataset_test,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
    shuffle=False,
)

# Display image and label
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Load in pretrained weights on imagenet
model = resnet50(ResNet50_Weights.IMAGENET1K_V1)

# Making new model
device = "cuda"
model = TransferLearningResNet().to(device)

writer = SummaryWriter("ResNetTesting")
current_epoch = 0

learning_rate = 1e-3
epochs = 50

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0, weight_decay=1e-6
)


# Training
def train_one_epoch(epoch_index, tb_writer):
    # Will add to running_loss each time
    running_loss = 0.0
    last_loss = 0.0
    for batch, (X, y) in enumerate(tqdm(train_dataloader, leave=False)):
        # Make sure predictions and labels are on same device
        X = X.to(device)
        y = y.to(device)

        # Zero gradients (reset) for every batch
        optimizer.zero_grad()

        # Make predictions
        X = X.permute(0, 3, 1, 2)
        pred = model(X)
        # Compute loss
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        loss.backward()
        # Adjust weights
        optimizer.step()

        # Get losses and report to tensorboard
        running_loss += loss.item()
        # Report losses every batch
        if batch % 8 == 7:
            last_loss = running_loss / 8
            tb_x = epoch_index * len(train_dataloader) + batch + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            # Reset running_loss for next batch
            running_loss = 0.0

    return last_loss


for epoch in range(epochs):
    print(f"Epoch{current_epoch + 1}")
    # Train
    model.train(True)
    # Get the average loss (running_loss/samples in batch)
    average_loss = train_one_epoch(current_epoch, writer)

    running_val_loss = 0.0
    # Arguments for AUC curve
    y_true = []
    y_score = []
    pred_all = []
    # Validation
    model.eval()
    # For evaluations, disable gradient computation
    with torch.no_grad():
        for batch, (X, y) in enumerate(valid_dataloader):
            # Make sure predictions and labels are on same device
            X = X.to(device)
            y = y.to(device)
            # Change order to match what it expects
            X = X.permute(0, 3, 1, 2)
            # Make predictions
            pred = model(X)
            # Convert to binary predictions
            binary_pred = pred.argmax(dim=1)
            # Append predition to list to compare to true values later
            pred_all.append(binary_pred)
            # Compute loss
            val_loss = loss_fn(pred.squeeze(), y)
            # Get losses
            running_val_loss += val_loss.item()
            # Get true labels and probabilities of predictions
            # Move to cpu to use numpy to make a list
            y_true.extend(y.cpu().numpy())
            y_score.extend(torch.softmax(pred, dim=1)[:, 1].cpu().numpy())

    average_val_loss = running_val_loss / (batch + 1)

    # Computing AUC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)

    print("Loss: train {} validation {}".format(average_loss, average_val_loss))
    # Log the average batch loss for training and validation
    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": average_loss, "Validation": average_val_loss},
        current_epoch + 1,
    )
    writer.add_scalar("AUC/validation", auc_score, current_epoch + 1)
    writer.flush()
    current_epoch += 1

writer.close()


# Testing
model.eval()
size = len(test_dataloader.dataset)
correct = 0
running_test_loss = 0.0
with torch.no_grad():
    for X, y in test_dataloader:
        # Make sure predictions and labels are on same device
        X = X.to(device)
        y = y.to(device)
        X = X.permute(0, 3, 1, 2)
        # Make predictions
        pred = model(X)
        # Compute loss
        test_loss = loss_fn(pred.squeeze(), y)
        # Get losses
        running_test_loss += test_loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Compute avg loss and accuracy
    average_test_loss = running_test_loss / len(test_dataloader)
    print(f"Average test loss: {average_test_loss:.6f}")

    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%")

# Making confusion matrix
# Make pred_all (tensors) comparable to y_true (list)
y_pred = []
for tensor in pred_all:
    pred_list = tensor.cpu().tolist()
    y_pred.extend(pred_list)
matrix = confusion_matrix(y_true, y_pred)
disp = sklearn.metrics.ConfusionMatrixDisplay(matrix)
disp.plot()
plt.show()

# Making ROC curve
RocCurveDisplay.from_predictions(y_true, y_score)
plt.show()
