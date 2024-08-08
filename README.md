# Quality Classifier Model
This repository contains a binary image quality classifier built using PyTorch. It categorizes extracted video frames of mice into good quality and bad quality. Good quality: can distinguish individual whiskers = clear image. Bad quality: face is blocked by the tail, or the image is dark and/or blurry. 

## Inputs
- assign variable project to PosixPath of the project folder
```python
# In main.py, ln 20:
project = Path("/path/to/project/folder")
```
- assign variable all_frames to a list of all PNG frames in the project folder
```python
# In main.py, ln 21:
all_frames = list(project.glob("path/to/frames.png"))
```
- assign variable writer to desired name for Tensorboard log folder
```python
# In main.py, ln 124:
writer = SummaryWriter("TensorboardFolderName")
```

## Outputs
Graphs to monitor model performance: 
- Loss/train
- AUC/validation
- Training vs. Validation loss
- Confusion Matrix
- ROC Curve

## Prerequisites
Please install the following libraries:
```bash
pip install -r requirements.txt
```
- ‘matplotlib’ version 3.8.4
- ‘numpy’ version 1.26.4
- ‘Pillow’ version 10.3.0
- ‘scikit-learn’ version 1.4.2
- ‘scikit-image’ version 0.23.2
- ‘tensorboard’ version 2.16.2
- ‘torch’ version 2.3.0
- ‘torchvision’ version 0.18.0
- ‘tqdm' version 4.66.2

## Project Organization
#### Quality-Classifier-Model/
- README.md (information for users)
- conv_layers.py (script for previous convolutional layer model) 
- image_quality_dataset.py (script for dataset handling)
- linear_layers.py (script for previous linear layer model)
- main.py (script for best ResNet-50 model)
- requirements.txt	(installations needed)
- transfer_learning_resnet.py (script for ResNet-50 model definition)

## Workflow
From a raw video of a freely moving mouse:
1. To track the location of the mouse face via labels on the nose, eyes, and ears: train a DeepLabCut model to label facial features on the dataset and run deeplabcut.analyze_videos() https://github.com/DeepLabCut/DeepLabCut 
2. To extract least blurry frames cropped around the mouse side profile: run raw_video_processing – extract_frames() https://github.com/A-Telfer/mouse-facial-expressions-2023/blob/main/mouse_facial_expressions/data/raw_video_processing.py
3. To categorize extracted frames by quality: run Quality-Classifier-Model
