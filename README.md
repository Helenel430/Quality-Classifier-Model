# Quality Classifier Model
This repository contains a binary image quality classifier built using PyTorch. It categorizes extracted video frames of mice into good quality and bad quality. Good quality: can distinguish individual whiskers = clear image. Bad quality: face is blocked by the tail, or the image is dark and/or blurry. 

## Inputs
All images organization: project_folder/folder_per_recording/PNG images 
- assign variable project to PosixPath of the project folder
```python
# In main.py, ln 20, and evaluation.py, ln 14: 
project = Path("/path/to/project/folder")
```
- assign variable images to list of all PNG frames
```python
# In evaluation.py, ln 16: 
images = list(project.glob("path/to/all_frames*.png"))
```
- replace FileName with desired name of csv file 
```python
# In evaluation.py, ln 46:
df.to_csv("FileName.csv", index=False)
```
Manually classified images for training organization: project_folder/sampled_frames_folder/good_frames and bad_frames folder/PNG images
- assign variable sampled_frames to list of all PNG frames under good_frames and bad_frames
```python
# In main.py, ln 21:
sampled_frames = list(project.glob("sampled_frames_folder/*/*.png"))
```
- replace FolderName with desired name for Tensorboard log folder
```python
# In main.py, ln 124:
writer = SummaryWriter("FolderName")
```

## Outputs
- csv file with path to image, model's class prediction, and associated probability
- Loss/train graph 
- AUC/validation graph
- Training vs. Validation loss graph
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
- conv_layers.py (previous convolutional layer model)
- evaluation.py (saving model predictions to csv file)
- image_quality_dataset.py (dataset handling)
- linear_layers.py (previous linear layer model)
- main.py (best ResNet-50 model)
- requirements.txt	(installations needed)
- transfer_learning_resnet.py (ResNet-50 model definition)

## Workflow
From a raw video of a freely moving mouse:
1. To track the location of the mouse face via labels on the nose, eyes, and ears: train a DeepLabCut model to label facial features on the dataset and run deeplabcut.analyze_videos() https://github.com/DeepLabCut/DeepLabCut 
2. To extract least blurry frames cropped around the mouse side profile: run raw_video_processing – extract_frames() https://github.com/A-Telfer/mouse-facial-expressions-2023/blob/main/mouse_facial_expressions/data/raw_video_processing.py
3. To categorize extracted frames by quality: run Quality-Classifier-Model
