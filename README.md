# Quality Classifier Model
This repository contains a binary image quality classifier built using PyTorch. It categorizes extracted video frames of mice into good quality and bad quality. Good quality: can distinguish individual whiskers = clear image. Bad quality: face is blocked by the tail, or the image is dark and/or blurry. 

## Inputs
- assign variable project to PosixPath of the project folder 
- assign variable all_frames to a list of all PNG frames in the project folder 
- assign variable writer to desired name for Tensorboard log folder 

## Outputs
Graphs to monitor model performance: 
- Loss/train
- AUC/validation
- Training vs. Validation loss
- Confusion Matrix
- ROC Curve

## Prerequisites
Please ensure you have the following libraries installed: 
- ‘matplotlib’
- ‘numpy’
- ‘Pillow’
- ‘scikit-learn’
- ‘scikit-image’
- ‘tensorboard’
- ‘torch’
- ‘torchvision’
- ‘tqdm’

## Project Organization
#### Quality-Classifier-Model/
- README.md (information for users) 
- image_quality_dataset.py (script for dataset handling) 
- main.py (script for training and testing loops)
- requirements.txt	(installations needed)
- transfer_learning_resnet.py (script for ResNet model definition)

## Workflow
From a raw video of a freely moving mouse:
1. To track the location of the mouse face via labels on the nose, eyes, and ears: train a DeepLabCut model to label facial features on the dataset and run deeplabcut.analyze_videos() https://github.com/DeepLabCut/DeepLabCut 
2. To extract least blurry frames cropped around the mouse side profile: run raw_video_processing – extract_frames() https://github.com/A-Telfer/mouse-facial-expressions-2023/blob/main/mouse_facial_expressions/data/raw_video_processing.py
3. To categorize extracted frames by quality: run Quality-Classifier-Model
