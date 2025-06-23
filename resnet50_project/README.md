# 🌿 Cassava Leaf Disease Classification with ResNet50 🌿

This project implements a ResNet50-based Convolutional Neural Network (CNN) to classify cassava leaf diseases using transfer learning.  
The model is trained and fine-tuned on the [Cassava Leaf Disease Classification dataset from Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 🚀 Project Overview

Cassava is an important crop in tropical regions but is vulnerable to several diseases affecting yield and quality.  
This project aims to build a robust model for early and accurate disease detection by leveraging the power of the ResNet50 architecture pretrained on ImageNet.

The approach involves:

- Using transfer learning from a pretrained ResNet50 model  
- Applying data augmentation for better generalization  
- Fine-tuning the classifier head for cassava disease categories  
- Monitoring training with metrics such as loss, accuracy, and macro F1-score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 📁 Folder Structure

resnet50_project/
├── resnet50_training.py # Training script for ResNet50 CNN model
├── plots/ # Training and validation plots (loss, accuracy, F1-score)
└── README.md # This documentation file

markdown
Копировать
Редактировать

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## ⚙️ Requirements

- Python 3.x  
- torch  
- torchvision  
- scikit-learn  
- pandas  
- matplotlib  
- Pillow  

Install dependencies with:

```bash
pip install torch torchvision scikit-learn pandas matplotlib pillow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

▶️ Usage
Run the training script to start model training and evaluation:

bash
Копировать
Редактировать
python resnet50_training.py
The script outputs training and validation losses, accuracy, and macro F1 scores per epoch.
The best model weights will be saved as best_model.pth.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Results
Best validation accuracy: 79%

Best validation macro F1 score: 67%

Training and validation losses decreased steadily over epochs, indicating effective learning.

Training and validation metrics are visualized in the plots/ folder.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔮 Future Improvements
Add confusion matrix and class-wise precision/recall reporting

Experiment with hyperparameter tuning and alternative architectures

Include inference script and pretrained model weights

Evaluate model on unseen test data

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 Dataset
Data used in this project comes from the Cassava Leaf Disease Classification Kaggle competition:
https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview
