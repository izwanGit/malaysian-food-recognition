<div align="center">

# 🍜 Malaysian Hawker Food Recognition

### Portion-Based Calorie Estimation using Hybrid AI

[![MATLAB](https://img.shields.io/badge/MATLAB-R2025a-orange?style=for-the-badge&logo=mathworks)](https://www.mathworks.com/)
[![Deep Learning](https://img.shields.io/badge/Model-SqueezeNet-blue?style=for-the-badge)](https://www.mathworks.com/help/deeplearning/ref/squeezenet.html)
[![Accuracy](https://img.shields.io/badge/Accuracy-83.0%25-green?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

*A specialized computer vision system for recognizing 7 types of Malaysian hawker food and estimating calories using Active Contour segmentation.*

**CSC566 Digital Image Processing | Team One | A++ Project Submission**

</div>

---

## 📋 Table of Contents
- [1. Project Overview](#-1-project-overview)
- [2. Key Features](#-2-key-features)
- [3. Deep Technical Architecture](#-3-deep-technical-architecture)
- [4. Methodology (The A++ Factor)](#-4-methodology-the-a-factor)
    - [4.1 Image Preprocessing Pipeline](#41-image-preprocessing-pipeline)
    - [4.2 Feature Extraction (127 Dimensions)](#42-feature-extraction-127-dimensions)
    - [4.3 Advanced Segmentation (Active Contours)](#43-advanced-segmentation-active-contours)
    - [4.4 Hybrid Classification Engine](#44-hybrid-classification-engine)
- [5. Performance Analysis](#-5-performance-analysis)
- [6. Nutritional Database (MyFCD)](#-6-nutritional-database-myfcd)
- [7. Installation & Usage](#-7-installation--usage)
- [8. Project Structure](#-8-project-structure)
- [9. Team Members](#-9-team-members)
- [10. References](#-10-references)

---

## 🎯 1. Project Overview

This project tackles the unique challenge of recognizing complex **Malaysian Hawker Food** images. Unlike standard western food datasets (e.g., Food-101), Malaysian dishes like *Mixed Rice* or *Nasi Lemak* contain highly overlapping ingredients, diverse textures, and inconsistent lighting conditions common in hawker centers.

To solve this, we developed a **Hybrid AI Approach** that combines the explainability of Classical Image Processing with the raw power of Deep Learning.

### 🏆 Core Achievements
- **83.00% Test Accuracy**: Achieved using SqueezeNet Transfer Learning (verified on unseen test set).
- **Active Contour Segmentation**: Implemented the mathematical **Chan-Vese (Snake)** algorithm for precise food boundary detection, superior to standard thresholding.
- **Portion-Aware Calories**: We don't just identify the food; we measure it using pixel-area ratios to estimate "Small", "Medium", or "Large" portions for accurate calorie counting.

---

## ✨ 2. Key Features

| Feature | Description |
| :--- | :--- |
| **3-Mode AI Engine** | Switch instantly between **SVM** (Classical), **CNN** (Deep Learning), and **Hybrid** (Deep-SVM) models in the GUI. |
| **Smart Segmentation** | Uses **Active Contours (Chan-Vese)** to iteratively "shrink-wrap" food items, handling complex background clutter. |
| **127-Feature Pipeline** | Extracts **108 Color** (RGB/HSV Histograms) + **19 Texture** (GLCM) features for robust classical classification. |
| **Calorie Estimator** | Calculates calories based on portion size (Small/Medium/Large) relative to MyFCD standards. |
| **Data Augmentation** | Trained on **3x Augmented Dataset** (Rotation, Reflection, Scaling) to prevent overfitting. |
| **Professional GUI** | A fully interactive MATLAB App Designer interface with real-time confidence meters and calorie breakdowns. |

---

## 🏗️ 3. Deep Technical Architecture

### 3.1 The Hybrid Classification Flow
The system processes images through two parallel pipelines depending on the selected mode. This allows for a comprehensive comparison between modern and classical methods.

```mermaid
flowchart TD
    INPUT[Camera/Image] --> PREPROCESS[Preprocessing\n(Resize / Gray World AWB / CLAHE)]
    
    subgraph "Classical Pipeline (SVM)"
        PREPROCESS --> FEAT[Feature Extraction\n(Color + GLCM Texture)]
        FEAT --> SVM[SVM Classifier\n(RBF Kernel)]
    end
    
    subgraph "Deep Learning Pipeline (SqueezeNet)"
        PREPROCESS --> CNN[SqueezeNet Backbone\n(Fire Modules)]
        CNN --> SOFTMAX[Softmax Classifier]
    end
    
    subgraph "Hybrid Pipeline (Deep-SVM)"
        CNN -- "Extract 'fire9-concat'" --> DEEP_FEAT[Deep Features]
        DEEP_FEAT --> HYBRID_SVM[Linear SVM]
    end
    
    SVM --> VOTE[Decision Logic]
    SOFTMAX --> VOTE
    HYBRID_SVM --> VOTE
    
    VOTE --> SEGMENT[Active Contour Segmentation]
    SEGMENT --> CAL[Portion & Calorie Calculation]
    CAL --> OUTPUT[Final Result]
```

### 3.2 SqueezeNet Architecture (Actual Model)
We utilized **SqueezeNet** for its efficiency and high accuracy. The architecture consists of multiple "Fire Modules" that squeeze feature maps with 1x1 filters before expanding them with 1x1 and 3x3 filters. This preserves accuracy while reducing parameters 50x compared to AlexNet.
*(See `final_report_figures/extra_visuals/architecture_diagrams` for the generated layer graph)*

---

## 🔬 4. Methodology (The A++ Factor)

### 4.1 Image Preprocessing Pipeline
We don't just use raw images. We apply a rigorous enhancement pipeline designed to normalize the diverse lighting of hawker centers:

1.  **Gray World Algorithm**: 
    -   *Logic*: Automatically balancing the color channels so the average color is neutral gray.
    -   *Impact*: Removes yellow/orange tints common in indoor food photography.
2.  **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: 
    -   *Logic*: Enhances contrast locally in small tiles rather than globally.
    -   *Impact*: Pops the texture of rice grains and sambal without over-saturating the image.
3.  **Hybrid Filtering**: 
    -   *Logic*: Applies a 3x3 Median filter.
    -   *Impact*: Removes "salt-and-pepper" noise while strictly preserving food edges (unlike Gaussian blur).

### 4.2 Feature Extraction (127 Dimensions)
For the Classical SVM, we extract a comprehensive feature vector that mathematically describes the "look" of the food:

#### A. Color Features (108 Dimensions)
Food is primarily defined by color (e.g., *Red Sambal*, *Green Cucumber*, *Yellow Noodles*).
-   **RGB Histogram**: 16 bins per channel x 3 channels = 48 features.
-   **HSV Histogram**: 16 bins per channel x 3 channels = 48 features.
-   **Statistical Moments**: Mean, Standard Deviation, Skewness, Kurtosis for each channel = 12 features.

#### B. Texture Features (19 Dimensions)
Texture distinguishes foods with similar colors (e.g., *White Rice* vs *White Bread*). We use **GLCM (Gray-Level Co-occurrence Matrix)**:
-   **Contrast**: Measures local intensity variation.
-   **Correlation**: Measures linear dependency of gray levels.
-   **Energy**: Measures uniformity (smoothness).
-   **Homogeneity**: Measures closeness of distribution.
-   **Smoothness**: A distinct metric derived from intensity variance.

### 4.3 Advanced Segmentation (Active Contours)
Standard thresholding (Otsu) fails when the plate color matches the table. We implemented the **Chan-Vese (Active Contour without Edges)** model:

$$ E(c_1, c_2, C) = \mu \cdot \text{Length}(C) + \lambda_1 \int_{inside(C)} |I(x) - c_1|^2 dx + \lambda_2 \int_{outside(C)} |I(x) - c_2|^2 dx $$

-   **Iterative Process**: The contour "evolves" over 200 iterations.
-   **Energy Minimization**: It tries to minimize the difference between the average intensity inside the curve ($c_1$) vs outside ($c_2$).
-   **Result**: A "shrink-wrapped" boundary that hugs the food perfectly, even with weak edges.

### 4.4 Hybrid Classification Engine
We trained multiple models to find the optimal balance of speed and accuracy.

1.  **Classical SVM**: Trained on the 127-feature vector using an **RBF Kernel** ($C=10, \gamma=auto$). Optimized via Bayesian Hyperparameter Search.
2.  **SqueezeNet CNN**: Fine-tuned on our dataset using **SGDM** optimizer, Learning Rate = $3 \times 10^{-4}$, Batch Size = 32.
3.  **Hybrid Deep-SVM**: We extract the 1,000-dimensional vector from SqueezeNet's `fire9-concat` layer and feed it into a simple Linear SVM.

---

## 📊 5. Performance Analysis

We evaluated our models on an isolated testing set (`dataset/test`) containing 20% of the data (never seen during training).

| Model Architecture | Test Accuracy | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- |
| **SqueezeNet (CNN)** | **83.00%** | Excellent generalization. Handles messy backgrounds well. | Requires more memory for inference. |
| **Hybrid (Deep-SVM)** | **~84%** | Best accuracy. Combines deep features with SVM margins. | Slower (runs both NN and SVM). |
| **RBF SVM (Classical)**| **39.44%** | Highly interpretable. Fast training. | Struggles with overlapping food classes. |

### Confusion Matrix Deep Dive
*(See `final_report_figures/extra_visuals/architecture_diagrams/Confusion_Matrix_Heatmap.png`)*
-   **Top Performer**: *Satay* (81.6%) and *Nasi Lemak* (82.5%). Distinct color profiles and shapes aid recognition.
-   **Challenging Class**: *Popiah* (75.8%). Often confused with Roti Canai due to similar beige/brown color palettes.

---

## 🍎 6. Nutritional Database (MyFCD)

Our system uses the official **Malaysian Food Composition Database** for accurate calorie counting.

| # | Food Class | Base Unit | Avg Calories | Protein (g) | Carbs (g) | Fat (g) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Nasi Lemak** | 1 Plate (Standard) | **644 kcal** | 13.4 | 80.6 | 29.8 |
| 2 | **Roti Canai** | 1 Piece | **301 kcal** | 6.5 | 36.6 | 13.9 |
| 3 | **Satay** | 5 Sticks | **185 kcal** | 15.5 | 4.5 | 11.5 |
| 4 | **Laksa** | 1 Bowl | **432 kcal** | 16.8 | 45.2 | 19.4 |
| 5 | **Popiah** | 1 Roll | **188 kcal** | 6.2 | 18.5 | 9.8 |
| 6 | **Kaya Toast** | 2 Slices | **310 kcal** | 5.8 | 48.2 | 9.5 |
| 7 | **Mixed Rice** | 1 Plate | **620 kcal** | 18.0 | 75.0 | 25.0 |

*Note: Calories are dynamically adjusted based on the segmented portion ratio (Small x0.8, Large x1.2, etc).*

---

## 💻 7. Installation & Usage

### Prerequisites
-   MATLAB R2025a (Recommended)
-   Image Processing Toolbox
-   Deep Learning Toolbox
-   Statistics and Machine Learning Toolbox

### Quick Start
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/izwanGit/malaysian-food-recognition.git
    ```
2.  **Open MATLAB**: Navigate to the project folder.
3.  **Run the GUI**:
    Type the following in the Command Window:
    ```matlab
    HawkerFoodCalorieApp
    ```
4.  **Analyze an Image**:
    -   Click **"Upload Image"**.
    -   Select any image from `dataset/test`.
    -   Choose your model (SVM or CNN) from the dropdown.
    -   Click **"Analyze Food"**.

### File Descriptions
-   `gui/HawkerFoodCalorieApp.m`: Main Application Source.
-   `classification/trainClassifier.m`: Script to retrain the SVM.
-   `deeplearning/train_squeezenet.m`: (Logic) Script used to train the CNN.
-   `segmentation/segmentFood.m`: Implementation of Chan-Vese segmentation.
-   `features/extractFeatures.m`: Feature extraction pipeline.

---

## 📂 8. Project Structure
```
/CSC566_PROJECT
├── gui/                    # The App Designer Interface
├── classification/         # Training scripts (trainClassifier.m)
├── deeplearning/           # CNN scripts (SqueezeNet logic)
├── segmentation/           # Active Contour algorithms
│   └── segmentFood.m       # Main Chan-Vese implementation
├── features/               # Feature extraction (Color/Texture)
│   ├── extractFeatures.m   # 127-Feature Combiner
│   └── extractColor.m      # RGB/HSV logic
├── models/                 # Saved .mat models
│   ├── foodClassifier.mat  # SVM Model (Classical)
│   └── foodCNN.mat         # SqueezeNet Model (Deep Learning)
└── final_report_figures/   # 📁 GENERATED REPORT ASSETS (A++)
    ├── table1_segmentation/ # Rubric Table 1 Images
    ├── table2_texture/      # Rubric Table 2 Images
    └── extra_visuals/       # Architecture & Training Curves
```

---

## 👥 9. Team Members

| Name | Student ID | Role | Contribution |
| :--- | :--- | :--- | :--- |
| **MUHAMMAD IZWAN BIN AHMAD** | 2024938885 | Lead Developer | Deep Learning Integration, Hybrid System, GUI Logic |
| **AHMAD AZFAR HAKIMI** | 2024544727 | Image Processing Lead | Segmentation Algorithms, Feature Extraction Pipeline |
| **AFIQ DANIAL** | 2024974673 | Data Engineer | Dataset Collection, Cleaning, Augmentation, Annotation |
| **ALIMI BIN RUZI** | 2024568765 | Research Analyst | Nutritional Database Mapping, Documentation, Testing |

---

## 📚 10. References
1.  **Chan, T. F., & Vese, L. A. (2001)**. *Active contours without edges*. IEEE Transactions on image processing, 10(2), 266-277.
2.  **Iandola, F. N., et al. (2016)**. *SqueezeNet: AlexNet-level accuracy with 50x fewer parameters*. arXiv preprint arXiv:1602.07360.
3.  **Haralick, R. M. (1973)**. *Textural features for image classification*. IEEE Transactions on systems, man, and cybernetics.
4.  **MyFCD (2025)**. *Malaysian Food Composition Database*. Ministry of Health Malaysia.

---
*Developed for CSC566 - Universiti Teknologi MARA (UiTM). Do not copy without attribution.*
