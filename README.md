# Using Machine Learning to estimate calories

This project uses Machine Learning to estimate the number of calories in a food item.

## Files

| File                  | Description                                                                       |
|-----------------------|-----------------------------------------------------------------------------------|
| [main.py](main.py)               | Contains the main application and the calorie estimation component                |
| [volume.py](volume.py)             | Contains the volume estimation component and all functions related to it          |
| [model.py](model.py)           | Contains the image classification model and all functions related to it           |
| [image_segmentation.py](image_segmentation.py) | Contains the image segmentation model and all functions related to it             |
| [helpers.py](helpers.py)            | Contains helper functions - Change `DebugMode` to `False` to disable debug output |
| [CameraWidget.py](CameraWidget.py) | QWidget that deals with the user's webcam information |
| [ThumbWidget.py](ThumbWidget.py)  | QWidget that allows the user to change thumb measurements |

## Requirements
This project requires [Anaconda3](https://www.anaconda.com/products/individual) and the following libraries to run:
 ```console
 OpenCV2
 PyTorch
 TorchVision
 GitLFS
  ```
  
## Usage


1. (Windows) Open Anaconda Prompt and install the libraries
   ```bash
   conda install -c menpo opencv
   conda install -c pytorch pytorch torchvision
   conda install -c conda-forge git-lfs
   ```

2. Clone the repo:
    ```bash
    git clone https://github.com/Yarmeli/FinalYearProject.git
    ```
 
3. Run the main application:
   ```bash
   cd FinalYearProject
   python main.py
   ```