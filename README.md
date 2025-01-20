# Chronic Venous Insufficiency Classification Project

This project is aimed at developing a classification model for Chronic Venous Insufficiency (CVI) using deep learning techniques. The code leverages PyTorch and related libraries for building, training, and evaluating the model.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Models and Transformations](#models-and-transformations)
- [Loss Functions](#loss-functions)
- [Training](#training)
- [Evaluation](#evaluation)

## Features

1. **Custom Data Transformations**: Includes preprocessing techniques like edge detection, contrast adjustment, and multi-channel input processing.
2. **Loss Functions**: Multiple loss functions such as Dice Loss, Kappa Loss, Tversky Loss, and Label Smoothing Cross-Entropy.
3. **Model Support**: Works with pre-trained models like EfficientNet and ResNet, with options to modify the input layers for additional channels.
4. **Evaluation Metrics**: Includes F1 score and Cohen's Kappa for comprehensive model evaluation.
5. **Cross-Validation**: Uses multiple seeds to train and evaluate models, ensuring robustness.

## Requirements

- Python 3.12+
- PyTorch 2.0+
- torchvision 0.13+
- numpy
- scikit-learn
- OpenCV
- tqdm
- vit-pytorch

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Murattut/Chronic-Venous-Insufficiency.git
   cd Chronic-Venous-Insufficiency
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that your dataset is placed in the `dataset/` directory or update the path in the code.

## Usage

### Training
To train the model, simply run:
```bash
python train.py
```
This script will load the dataset, apply transformations, and train the model.

### Dataset Structure
The dataset should follow the ImageFolder format:
```
/dataset/
    /class_1/
    /class_2/
    /class_3/
```

### Configuration
Update the following parameters in the script as needed:
- `data_dir`: Path to the dataset directory.
- `num_epochs`: Number of training epochs.
- `batch_size`: Batch size for training.

## Models and Transformations

- **EfficientNet and ResNet**: Pre-trained models with modified first layers for additional channels.
- **Custom Transformations**: The `CustomTransform` class handles preprocessing steps like CLAHE, GLCM, and multi-channel augmentation.

## Loss Functions
The project includes several loss functions tailored for imbalanced data and segmentation tasks:
- Dice Loss
- Tversky Loss
- Kappa Loss
- Label Smoothing Cross-Entropy

These can be easily switched in the `train` function.

## Training
The training process involves:
1. Loading the dataset with data augmentations.
2. Modifying the model's input layers to handle multi-channel data.
3. Training the model using Adam optimizer and Cross-Entropy loss (or a custom loss).
4. Evaluating on a validation set using metrics like F1 Score and Cohen's Kappa.

## Evaluation
The best-performing model is selected based on test accuracy. Evaluation metrics include:
- Test Accuracy
- F1 Score (Weighted)
- Cohen's Kappa

## Contribution
Feel free to submit issues or pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
