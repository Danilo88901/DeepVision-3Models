# 🌿 Cassava Leaf Disease Classification with ConvNeXt + Grad-CAM

This project focuses on the classification of cassava leaf diseases using various deep learning architectures. It builds upon previous work involving basic CNNs, expanding the scope with state-of-the-art pretrained models and interpretability techniques such as **Grad-CAM**.

---

## 📌 Dataset

- We used the **Cassava Leaf Disease Classification** dataset from [Kaggle](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview).
- The dataset contains **21,397 images** of cassava leaves labeled into 5 disease categories.

---

## 🧠 Models Compared

We trained and evaluated the following models:

| Model                         | Best Accuracy | Best F1 Score |
|------------------------------|---------------|----------------|
| 🧬 ConvNeXt-Tiny (pretrained) | **86.36%**     | **76.48%**      |
| 🧪 Custom CNN with Inception  | 53.46%         | 43.05%          |
| 🏛️ ResNet-50 (pretrained)     | 79.07%         | 67.18%          |

---

## 🔍 Grad-CAM Visualization

To interpret model decisions, we applied **Grad-CAM** to visualize which parts of the cassava leaves influenced the model's predictions. This was particularly used with the ConvNeXt-Tiny model.

> Grad-CAM helps in understanding which regions in the input image were important for the model’s decision.

![Example CAM Visualization](example_gradcam.png) <!-- Optional: Replace with your own Grad-CAM output -->

---

## ⚙️ Training Setup

- Trained for **12 epochs** on all models.
- Used **AdamW** optimizer with:
  - Learning rate: `4e-5`
  - Weight decay: `1e-3`
- Scheduler: `StepLR`, decreasing LR every 5 epochs (gamma=0.1).
- Loss function: `CrossEntropyLoss` with class weights to handle class imbalance.
- Data augmentations:
  - Random horizontal & vertical flip
  - Color jitter
  - Random rotation
  - Resize to (224x224)
  - Normalization (ImageNet)

---

## 📈 Evaluation Strategy

- The dataset was split into **90% training** and **10% validation**.
- Metrics logged for every epoch:
  - **Loss**
  - **Accuracy**
  - **F1 Score (Macro)**
- Models were saved based on best validation **F1 Score**.

---

## 🔬 Difference from Previous Project

- **Before:** A basic CNN was built and compared to a pretrained ResNet-50. No interpretability tools were used.
- **Now:**
  - Added **ConvNeXt-Tiny**, a modern architecture.
  - Used **Grad-CAM** for interpretability.
  - More advanced data augmentations.
  - Larger dataset (~21k images).
  - Focused not only on accuracy but also on **F1 Score**, which is crucial for imbalanced data.

---

## 🧾 Dependencies

- Python 3.10+
- PyTorch
- torchvision
- scikit-learn
- OpenCV
- Matplotlib
- PIL (Pillow)

---

## ✅ Results Summary

ConvNeXt-Tiny outperformed all other models both in accuracy and F1 score. Here's a quick recap:

- 🔝 **ConvNeXt-Tiny**
  - Accuracy: **86.36%**
  - F1 Score: **76.48%**

- 🧪 **Custom CNN (with Inception block)**
  - Accuracy: **53.46%**
  - F1 Score: **43.05%**

- 🏛️ **ResNet-50**
  - Accuracy: **79.07%**
  - F1 Score: **67.18%**

Grad-CAM visualizations further validated that the model focuses on disease-affected leaf areas, confirming that it learned meaningful representations.

---

> For any further questions or visualization examples, feel free to explore the Jupyter notebooks and scripts included in this repo.
