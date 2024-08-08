# importing necessary libraries
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn
import torch
from PIL import Image
from skimage.io import imread
from sklearn.metrics import RocCurveDisplay, auc, confusion_matrix, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


class ImageDataset(Dataset):
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
        "making sure List of Dict has image 'path' and int 'target' that is 0 or 1"
        sample = self.image_map[idx]
        assert "path" in sample, "Each sample must have a path property"
        assert "target" in sample, "Each sample must have a target property"
        image_path = sample["path"]
        image = self.read_image(image_path)
        target = sample["target"]
        assert isinstance(target, int), "Your target must be an integer"
        assert target in [0, 1], "Your target must be 0 or 1"

        # Scale the image to size (32, 32)
        image = Image.fromarray(image)
        image = image.resize((32, 32))
        image = np.array(image)

        # Change from bytes [0,255] to a float [0,1]
        image = image / 255
        image = image.astype(np.float32)

        return image, target


# Defining paths
project = Path("/home/brennamacaulay/Desktop/Stress_Facial_Exp-Helene-2024-03-20")
all_frames = list(project.glob("sampled_frames_folder/*/*.png"))

# Making the training (70%) and testing datasets (10%)
# The length of dataset may not perfectly fit into these numbers
total_count = len(all_frames)
train_count = int(0.7 * total_count)
test_count = int(0.1 * total_count)
# Roughly 20% will be validation data
valid_count = total_count - train_count - test_count

# Ensures reproducibility
generator = torch.Generator().manual_seed(42)

# Splitting the dataset
train_data, valid_data, test_data = random_split(
    all_frames, [train_count, valid_count, test_count], generator=generator
)

# Initalize List of Dictionaries for training and testing
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


# Testing
# Wrapping each list in the dataset
dataset_train = ImageDataset(image_map_train)
dataset_valid = ImageDataset(image_map_valid)
dataset_test = ImageDataset(image_map_test)

# Using dataloader to randomly load datasets
batch_size = 8

train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
# Do not shuffle validation and testing for evaluation consistency
valid_dataloader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Using {device} device")


# Defining network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32 * 32 * 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        "flatten the image, meaning turn it from a tensor into a vector"
        x = self.flatten(x)
        x = x.float()
        # Feed the flattened vector into nn.sequential()
        logits = self.linear_relu_stack(x)
        # Return the prediction, which is stored in logits
        return logits


model = NeuralNetwork().to(device)


writer = SummaryWriter("mylogdir")
current_epoch = 0

learning_rate = 1e-2
epochs = 50

# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
model = NeuralNetwork().to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=0.7, weight_decay=0.1
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
        pred = model(X)
        # Compute loss
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        loss.backward()
        # Adjust weights
        optimizer.step()

        # Get losses and report to tensorboard
        running_loss += loss.item()
        # report losses every batch
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

    # Input labels and predictions to get FPR and TPR
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
