# 🔍 Explainable AI (XAI) for CIFAR-10 Image Classification

This project focuses on moving beyond "black-box" machine learning by implementing Explainable AI (XAI) techniques on a Convolutional Neural Network (CNN). Using the CIFAR-10 dataset, we demonstrate how transparency tools like Grad-CAM and SHAP can be used to debug model behavior, ensure fairness, and build trust in AI decisions.

---

## 🎯 Objectives

- **Transparency:** Visualize the "why" behind model predictions.  
- **Interpretability:** Apply game-theory (SHAP) and gradient-based (Grad-CAM) methods to identify influential image regions.  
- **Performance:** Achieve robust classification accuracy while maintaining high explainability.  
- **Debugging:** Identify model biases (e.g., relying on background pixels rather than the object).  

---

## 📊 Dataset: CIFAR-10

The model is trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes:

- **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck  
- **Pre-processing:**  
  - Pixel normalization (0-1)  
  - One-Hot encoding of labels  
  - Data augmentation for better generalization  

---

## 🏗️ Model Architecture

A Convolutional Neural Network (CNN) was implemented using the Keras Functional API to allow for intermediate layer tapping (essential for Grad-CAM).

| Layer Type        | Configuration         | Purpose                          |
|------------------|----------------------|----------------------------------|
| Input            | (32, 32, 3)          | Standard CIFAR image size        |
| Conv2D + BN      | 32 filters, 3x3      | Low-level feature extraction     |
| MaxPooling       | 2x2                  | Spatial dimensionality reduction |
| Conv2D (Target)  | 64 filters, 3x3      | High-level features (Grad-CAM)   |
| Flatten & Dense  | 128 units, ReLU      | Non-linear reasoning             |
| Dropout          | 0.5                  | Prevent overfitting              |
| Output (Softmax) | 10 units             | Class probability prediction     |

---

## 💡 Explainable AI Techniques

### 1. Grad-CAM (Gradient-weighted Class Activation Mapping)

- **Type:** Local Explanation  
- **How it works:**  
  Uses gradients flowing into the final convolutional layer to generate a heatmap highlighting important regions.  
- **Insights:**  
  Helps determine whether the model focuses on the object (e.g., deer legs) or irrelevant background.

---

### 2. SHAP (SHapley Additive exPlanations)

- **Type:** Global/Local Hybrid (GradientExplainer)  
- **How it works:**  
  Based on cooperative game theory, assigns importance values to each pixel.  
- **Interpretation:**  
  - 🔴 Red Pixels → Increase prediction probability  
  - 🔵 Blue Pixels → Decrease prediction probability  

---

## 📈 Results & Visualizations
### Model Performance
**Accuracy:** ~XX% (Enter your test accuracy here)

**F1-Score:** ~X.XX (Enter your macro-avg F1 here)

XAI Analysis
!(Screenshot 2026-04-15 154734.png)
!()

Grad-CAM: Shows the model correctly identifies structural features (like wings for airplanes).

SHAP: Reveals pixel-level contributions to the final confidence score.

## 🛠️ Tools Used
**Deep Learning:** TensorFlow, Keras

**XAI:** SHAP, Custom Grad-CAM Implementation

**Data Science:** NumPy, Pandas, Matplotlib, Seaborn

**Environment:** kaggle

## 👨‍💻 Author
**Darshan Bhabad**
