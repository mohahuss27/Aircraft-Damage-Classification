# Aircraft Damage Classification with Keras

## Overview

This project builds a deep learning model to automatically classify aircraft surface damage into **cracks** and **dents** using Convolutional Neural Networks (CNNs) with transfer learning.

By leveraging a pre-trained VGG16 model, the system achieves strong performance on image classification tasks while reducing training time and data requirements.

## Dataset

The dataset used is the Aircraft Damage Dataset v1, which contains images of aircraft damage categorized into:
- **Cracks**: Images showing crack damage on aircraft surfaces
- **Dents**: Images showing dent damage on aircraft surfaces

The dataset is organized into three splits:
- `train/`: Training images
- `valid/`: Validation images  
- `test/`: Test images

Each split contains subdirectories for the two damage classes.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook

## Installation

Install the required packages using pip:

```bash
pip install tensorflow keras numpy matplotlib jupyter
```

## Model Architecture

The model uses transfer learning with **VGG16** as the base model:
- **Base Model**: Pre-trained VGG16 (ImageNet weights, top layers removed)
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layers**: 512 neurons with ReLU activation
- **Dropout**: 30% dropout for regularization
- **Output Layer**: Single neuron with sigmoid activation for binary classification

## Training Details

- **Optimizer**: Adam with learning rate 0.00005
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Data Augmentation**: Rotation, width/height shift, horizontal flip
- **Early Stopping**: Monitors validation loss with patience
- **Batch Size**: Configurable (default: 32)
- **Image Size**: 224x224 pixels

## Usage

1. Ensure the dataset is in the `aircraft_damage_dataset_v1/` directory
2. Open the notebook in Jupyter
3. Run cells sequentially from top to bottom
4. The notebook will:
   - Load and preprocess the data
   - Build and compile the model
   - Train the model with early stopping
   - Visualize training curves
   - Evaluate on test data

## Results

The notebook provides:
- Training and validation loss/accuracy curves
- Best model epoch identification
- Test set evaluation metrics
- Visual plots of model performance
- Validation Accuracy: **77%**  
- Test Accuracy: **84%**  
- Best Epoch: **20**  

## Business Value

This model can support automated aircraft inspection systems by:
- Reducing manual inspection time  
- Assisting maintenance engineers in detecting structural damage  
- Improving safety through early detection of cracks and dents
  
## Key Features

- **Transfer Learning**: Leverages pre-trained VGG16 for feature extraction
- **Data Augmentation**: Improves model generalization
- **Early Stopping**: Prevents overfitting
- **Visualization**: Comprehensive plots for model analysis
- **Binary Classification**: Distinguishes between crack and dent damage

## Challenges & Learnings

- Preventing overfitting with limited training data  
- Selecting effective data augmentation techniques  
- Balancing model complexity and generalization  
- Understanding the strengths and limitations of transfer learning  

## File Structure

```
aircraft_damage_classification/
├── Aircraft_Damage_Classification.ipynb  # Main notebook
├── aircraft_damage_dataset_v1/           # Dataset directory
│   ├── train/
│   │   ├── crack/
│   │   └── dent/
│   ├── valid/
│   │   ├── crack/
│   │   └── dent/
│   └── test/
│       ├── crack/
│       └── dent/
└── README.md                             # This file
```

## Notes

- The base VGG16 model is frozen during training to preserve pre-trained features
- Early stopping ensures optimal training duration
- The model is designed for binary classification of aircraft damage types
- All preprocessing follows VGG16 input requirements
