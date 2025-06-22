# DeepVision-3Models
Benchmarking three CNN architectures — ResNet-50, custom Inception-based model (Redmi), and ConvNeXt Tiny — on multi-class image classification tasks with integrated GRAM visualization for model interpretability. The project facilitates performance comparison through accuracy, F1-score, and loss metrics alongside visual attention analysis.
# RedmiNet-Compare: Benchmarking ResNet-50, Custom Inception, and ConvNeXt Tiny for Image Classification with GRAM Visualization

## Description

This project benchmarks three CNN architectures — ResNet-50, a custom Inception-based model (Redmi), and ConvNeXt Tiny — on a multi-class image classification task. It combines performance metrics (accuracy, F1-score, loss) with interpretability via Gradient-weighted Class Activation Mapping (GRAM), helping to understand how different models "see" images.

## Models Included

- **ResNet-50**: Residual network baseline with strong performance.
- **Custom Inception (Redmi)**: Enhanced Inception with increased width and depth.
- **ConvNeXt Tiny**: Modern CNN inspired by transformer architectures.

## Features

- Training and evaluation with consistent datasets.
- Metrics calculation: accuracy, macro F1-score, and loss.
- GRAM heatmap visualizations for model interpretability.
- Hyperparameter tuning capabilities.

## Requirements

- Python 3.7+  
- PyTorch 1.10+  
- torchvision  
- scikit-learn  
- matplotlib  

## Usage

1. Prepare dataset and dataloaders.
2. Initialize and train each model.
3. Evaluate and generate GRAM visualizations.
4. Compare models by metrics and attention maps.

## Contact

Author: D.Silienko  
Email: danya.kobal123456@gmail.com
GitHub: https://github.com/Danilo88901(Danilo88901)
