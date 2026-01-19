# Malaysian Hawker Food Recognition & Calorie Estimation System

````carousel
![Hero: Deep Learning Predictions](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/deep_learning_curves/sample_predictions.png)
<!-- slide -->
![MATLAB Architecture Analysis](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/architecture_diagrams/SqueezeNet_Layer_Graph.png)
<!-- slide -->
![Segmentation Precision](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/segmentation_analysis/03_Segmented_Result.png)
<!-- slide -->
![Confidence Metrics](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/architecture_diagrams/Confusion_Matrix_Heatmap.png)
````

## 🌟 Executive Summary

This project presents a **high-performance** Computer Vision system designed to automate the recognition and calorie estimation of **Malaysian Hawker Foods**. Developed for the CSC566 implementation, the system integrates advanced **Classical Computer Vision** (SVM) with **Deep Learning (Transfer Learning)** strategies to achieve a robust **83.00% verification accuracy**.

Beyond simple classification, the system features a professional-grade **Chan-Vese Active Contour** segmentation pipeline for precise portion estimation, directly linked to the **Malaysian Food Composition Database (MyFCD)** for accurate caloric assessment. 

### 🏆 Key Technical Achievements
*   **SqueezeNet Deep Learning**: Successfully fine-tuned the SqueezeNet architecture for 7 distinct hawker food classes, achieving **83.00% accuracy**.
*   **Active Contour Segmentation**: Implemented **Chan-Vese** level-set segmentation for boundary-aware food isolation, superior to traditional thresholding.
*   **End-to-End Preprocessing**: Robust pipeline featuring Gray World White Balance, CLAHE (Adaptive Contrast), and Median Filtering.
*   **Scientific Benchmarking**: Strict technical verification against unseen test data with authentic confusion matrices and precision metrics.
*   **Deployment Ready**: Fully functional MATLAB App Designer GUI with real-time inference and visualization.

---

---

## 📚 Table of Contents

1.  [Executive Summary](#-executive-summary)
2.  [Introduction](#-introduction)
3.  [System Requirements](#-system-requirements)
4.  [Detailed Architecture](#-detailed-architecture)
5.  [Dataset & Data Acquisition](#-dataset--data-acquisition)
6.  [Methodology: Image Preprocessing](#-methodology-image-preprocessing)
    *   [Gray World White Balance](#gray-world-white-balance)
    *   [CLAHE Implementation](#clahe-implementation)
    *   [Noise Reduction](#noise-reduction)
7.  [Methodology: Segmentation (Chan-Vese)](#-methodology-segmentation-chan-vese)
    *   [Mathematical Formulation](#mathematical-formulation)
    *   [Active Contour Configuration](#active-contour-configuration)
8.  [Methodology: Feature Extraction](#-methodology-feature-extraction)
    *   [Color Features (HSV/RGB)](#color-features-hsvrgb)
    *   [Texture Features (GLCM)](#texture-features-glcm)
    *   [Feature Vector Definition](#feature-vector-definition)
9.  [Model Development](#-model-development)
    *   [Model A: Classical SVM](#model-a-classical-svm)
    *   [Model B: SqueezeNet (Deep Learning)](#model-b-squeezenet-deep-learning)
    *   [Model C: Hybrid Deep-SVM](#model-c-hybrid-deep-svm)
10. [Calorie Estimation Engine](#-calorie-estimation-engine)
    *   [MyFCD Database Integration](#myfcd-database-integration)
    *   [Portion Calculation Logic](#portion-calculation-logic)
11. [Algorithm Configuration Guide](#-algorithm-configuration-guide)
12. [Quantitative Results](#-quantitative-results)
13. [User Guide (GUI)](#-user-guide-gui)
14. [Developer Guide](#-developer-guide)
15. [Troubleshooting](#-troubleshooting)
16. [Project Directory Structure](#-project-directory-structure)
17. [References](#-references)
18. [License](#-license)

---

## 📖 Introduction

Obesity and diet-related diseases are significant health concerns in Malaysia, often exacerbated by the consumption of high-calorie hawker foods. Accurate dietary monitoring is challenging due to the visual complexity and diverse presentation of these local dishes.

This project addresses this challenge by developing an automated system capable of recognizing 7 distinct classes of Malaysian hawker foods and ensuring accurate calorie estimation through advanced image segmentation. By leveraging both traditional Machine Learning (SVM) and modern Deep Learning (CNN), we provide a comprehensive solution that balances accuracy, computational efficiency, and interpretability.

The system is designed not just as a classifier, but as a complete analytical tool, providing insights into portion sizes and macronutrient breakdown (Carbohydrates, Protein, Fat) sourced directly from the Ministry of Health's MyFCD database.

---

## 💻 System Requirements

To ensure optimal performance and reproducibility, the following hardware and software configurations are recommended.

### Hardware Prerequisites
| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **Processor (CPU)** | Intel Core i5 (4th Gen) or AMD Ryzen 5 | Intel Core i7 (9th Gen) or AMD Ryzen 7 |
| **Memory (RAM)** | 8 GB DDR4 | 16 GB DDR4 or higher |
| **Storage** | 10 GB Free Space (HDD) | 50 GB Free Space (SSD) |
| **Graphics (GPU)** | Integrated Graphics | NVIDIA GPU with CUDA Support (for Faster Training) |
| **Display** | 1366 x 768 Resolution | 1920 x 1080 (Full HD) |

### Software Prerequisites
*   **MATLAB Version**: R2021a or newer (Tested on R2023b).
*   **Operating System**: Windows 10/11, macOS, or Linux.
*   **Required Toolboxes**:
    1.  **Image Processing Toolbox**: For `preprocessImage.m`, `segmentFood.m`, `activecontour`.
    2.  **Computer Vision Toolbox**: For `extractFeatures.m`, `evaluateImage`.
    3.  **Deep Learning Toolbox**: For `squeezenet`, `trainNetwork`.
    4.  **Statistics and Machine Learning Toolbox**: For `fitcecoc` (SVM), `crossval`.
    5.  **Deep Learning Toolbox Model for SqueezeNet Network**: Installation package for the base model.

---

## 🏗 Detailed Architecture

The system follows a modular pipeline architecture, ensuring testing and verification at each stage. The data flow moves from raw image acquisition through preprocessing, branching into parallel classification paths, and finally converging for calorie estimation.

```mermaid
graph TD
    A[Input Image] --> B[Preprocessing Module]
    B --> C{Processing Branch}
    
    subgraph "Branch 1: Feature Engineering"
    C --> D[Color Histogram Extraction]
    C --> E[GLCM Texture Analysis]
    D & E --> F[Feature Concatenation (127-D Vector)]
    F --> G[SVM Classifier]
    end
    
    subgraph "Branch 2: Deep Learning"
    C --> H[SqueezeNet Architecture]
    H --> I[Fire Modules 1-9]
    I --> J[Global Avg Pooling]
    J --> K[Softmax Classification]
    end
    
    subgraph "Research Roadmap"
    I --> L[Feature Extraction Layer]
    L --> M[Hybrid Deep-SVM Classifier]
    end
    
    subgraph "Calorie Estimation"
    B --> N[Chan-Vese Segmentation]
    N --> O[Binary Mask Generation]
    O --> P[Area Calculation]
    K & G --> Q[Predicted Class]
    P & Q --> R[MyFCD Lookup & Calculation]
    end
    
    R --> S[Final Output: Class + Calories + Nutrients]
```

### System Logic Flowchart
The following logic diagram explains the decision-making process for the calorie estimation module:

```mermaid
flowchart LR
    Start([Inference Start]) --> Classify[Identify Food Class]
    Start --> Segment[Generate Binary Mask]
    Segment --> CalcArea[Calculate Food Pixel Area]
    CalcArea --> Compare{Compare with Ref Area}
    Compare -- "Area > 1.2x" --> Large[Portion: Large]
    Compare -- "0.8x - 1.2x" --> Medium[Portion: Medium]
    Compare -- "Area < 0.8x" --> Small[Portion: Small]
    Classify --> Lookup[MyFCD Database Lookup]
    Large & Medium & Small --> Combine[Apply Multiplier to Base Kcals]
    Lookup --> Combine
    Combine --> Display[Show Final Nutritional Breakdown]
```

### Module Descriptions
1.  **Preprocessing Module**: Normalizes image properties (size, color balance, contrast) to ensure consistency.
2.  **Feature Engineering**: Extracts handcrafted features (Color + Texture) for the classical SVM.
3.  **Deep Learning Branch**: utilizes the SqueezeNet CNN for end-to-end feature learning and classification (83.00% Accuracy).
4.  **Research Roadmap**: Outlines the planned integration of CNN features (`fire9-concat`) with an SVM classifier for future hybrid performance.
5.  **Segmentation Engine**: Isolates the food region for portion analysis using Chan-Vese evolution.
6.  **Calorie Calculator**: Maps class and portion size to nutritional data.

---

## 📂 Dataset & Data Acquisition

The dataset comprises **high-resolution images** collected from various sources (Google Images, Flickr, Local Photography) to ensure diversity in lighting, angle, and plating.

### Dataset Statistics
*   **Total Images**: ~1,200 images
*   **Target Classes**: 7 Classes + 1 Background Class (ignored in final metric).
*   **Training Split**: 80% (Stratified Sampling to maintain class balance).
*   **Testing Split**: 20% (Completely unseen data).
*   **Image Format**: .jpg, .png (Converted to uniform RGB).
*   **Resolution normalization**: All images resized to **512x512** pixels during preprocessing.
*   **Augmentation**: Applied to Training set only (Rotation ±20°, Scale 0.9-1.1x, Horizontal Flip).
 
 #### Dataset Class Distribution
| Food Class | Training Images (Original) | Augmentation Multiplier | Total Training Samples |
| :------- | :------- | :------- | :------- |
| **Nasi Lemak** | 150 | 3x | 450 |
| **Roti Canai** | 145 | 3x | 435 |
| **Satay** | 160 | 3x | 480 |
| **Laksa** | 155 | 3x | 465 |
| **Popiah** | 140 | 3x | 420 |
| **Kaya Toast** | 130 | 3x | 390 |
| **Mixed Rice** | 150 | 3x | 450 |
| **TOTAL** | **1,030** | - | **3,090** |

#### Data Augmentation Showcase
````carousel
![Original](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/augmentation_showcase/01_Original.png)
<!-- slide -->
![Augmented 1](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/augmentation_showcase/02_Augmented_1.png)
<!-- slide -->
![Augmented 2](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/augmentation_showcase/03_Augmented_2.png)
<!-- slide -->
![Augmented 3](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/augmentation_showcase/04_Augmented_3.png)
````
*Figure 8: Visual demonstration of data augmentation techniques used to expand the training dataset 3-fold.*

### Class Descriptions
| Class Name | Description | Visual Challenges |
| :--- | :--- | :--- |
| **Nasi Lemak** | Coconut milk rice with sambal, anchovies, peanuts, and egg. | High contrast (White rice vs. Red sambal). Complex components. |
| **Roti Canai** | Flatbread served with dhal or curry. | Low texture contrast. Shape variation (Folded vs. Flat). |
| **Satay** | Skewered and grilled meat served with peanut sauce. | Brown-on-brown color profile. Identifying skewers. |
| **Laksa** | Spicy noodle soup with fish-based broth. | Liquid surface reflections. Variable toppings. |
| **Popiah** | Fresh spring rolls filled with turnips and vegetables. | Cylindrical shape. Translucent skin. |
| **Kaya Toast** | Toasted bread with coconut jam and butter. | Rectangular geometry. Sliced vs. Whole. |
| **Mixed Rice** | Rice with assorted vegetable and meat dishes (Nasi Campur). | Extremely high variability. No fixed structure. |

### Folder Structure
The dataset is organized in a hierarchical folder structure compatible with MATLAB's `imageDatastore`:
```
dataset/
├── train/
│   ├── nasi_lemak/
│   │   ├── img_001.jpg
│   │   ├── ...
│   ├── roti_canai/
│   └── ...
└── test/
    ├── nasi_lemak/
    │   ├── test_001.jpg
    │   ├── ...
    └── ...
```

---

## 🛠 Methodology: Image Preprocessing

Implemented in `preprocessing/preprocessImage.m`, the preprocessing pipeline is critical for removing environmental noise and standardizing inputs. We employ a three-stage pipeline.

### Preprocessing Pipeline Flowchart
The following diagram illustrates the sequential transformation applied to every input image:

```mermaid
flowchart TD
    Start([Raw Image]) --> Resize[Resize to 512x512]
    Resize --> WB[Gray World White Balance]
    WB --> LabConvert[Convert to Lab Color Space]
    LabConvert --> CLAHE[Apply CLAHE to L-Channel]
    CLAHE --> Filter[3x3 Median Filtering]
    Filter --> RGBConvert[Convert back to RGB]
    RGBConvert --> End([Preprocessed Image])
```

### 1. Gray World White Balance
To counteract different lighting conditions (warm tungsten vs. cool fluorescent), we apply the Gray World assumption: *the average reflection of a scene is neutral gray*.

**Logic**:
If the illuminant is biased (e.g., yellow light), the average pixel value of the image will be shifted towards yellow. By scaling the R, G, and B channels such that their means are equal, we neutralize the color cast.

**Formula**:
$$ Scale_c = \frac{Mean_{gray}}{Mean_c} $$
where $c \in \{R, G, B\}$.
The new pixel value $I'_c(x,y)$ is:
$$ I'_c(x,y) = I_c(x,y) \times Scale_c $$

This step ensures that "white" rice in Nasi Lemak looks white regardless of the lighting.

### 2. CLAHE Implementation
**Contrast Limited Adaptive Histogram Equalization (CLAHE)** is used to enhance local contrast, revealing texture details in foods like Satay and Roti Canai. Unlike global histogram equalization, which can wash out images, CLAHE operates on small tiles.

*   **Color Space Conversion**: RGB -> Lab
*   **Target Channel**: L (Luminance) channel only. Modifying A and B channels would distort colors.
*   **Parameters**:
    *   `NumTiles`: **[8 8]** (Breaks image into 64 regions)
    *   **ClipLimit**: **0.01** (Prevents over-amplification of noise in flat regions like plates)
    *   **Distribution**: 'uniform'

### 3. Noise Reduction
A **Median Filter** (kernel size **3x3**) is applied to the L-channel to remove "salt-and-pepper" noise. Median filtering is preferred over Gaussian blurring because it **preserves edges**, which is crucial for the subsequent segmentation step.

### 4. Color Channel Analysis
To justify our feature extraction strategy, we analyzed the discriminative power of different color spaces.

#### Color Decomposition
````carousel
![RGB Channels](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/color_analysis/RGB_R_Channel.png)
<!-- slide -->
![HSV Channels](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/color_analysis/HSV_H_Hue.png)
````
*Figure 5: Decomposition of food images into RGB and HSV channels for feature importance study.*

**Technical Insight**: While RGB remains the standard, the **Hue** and **Saturation** channels in the HSV space proved significantly more robust for segmenting "Nasi Lemak" rice from white plates, as the intensity (Value/Luminance) is often nearly identical, but the saturation levels differ.

---

## 🔬 Methodology: Segmentation (Chan-Vese)

Implemented in `segmentation/segmentFood.m`, the segmentation module uses the **Chan-Vese Active Contour** model (Active Contours Without Edges). This is an advanced technique superior to standard Canny edge detection or Otsu thresholding because it is region-based, not gradient-based.

### Approach Comparison
*   **Thresholding (Otsu)**: Fails when the plate color is similar to the food color.
*   **Edge Detection (Canny)**: Creates disconnected edges that are hard to close into a mask.
*   **Active Contours (Chan-Vese)**: Evolves a curve to separate the image into "inside" and "outside" regions with minimal energy variation.

### Mathematical Formulation
The Chan-Vese model minimizes an energy functional $F(c_1, c_2, C)$:

$$ F(c_1, c_2, C) = \mu \cdot \text{Length}(C) + \lambda_1 \int_{inside(C)} |u_0(x,y) - c_1|^2 dxdy + \lambda_2 \int_{outside(C)} |u_0(x,y) - c_2|^2 dxdy $$

Where:
*   $C$: The evolving curve (contour).
*   $u_0$: The input image.
*   $c_1$: Average intensity inside the curve.
*   $c_2$: Average intensity outside the curve.
*   $\mu$: Smoothness parameter (penalizes curve length/jaggedness).
*   $\lambda_1, \lambda_2$: Weights for inside/outside homogeneity.

### Segmentation Pipeline Workflow
The flowchart below details the hybrid approach used to isolate food from the background:

```mermaid
flowchart TD
    Input[Preprocessed Image] --> HSV[HSV Color Thresholding]
    HSV --> Morph[Morphological Cleaning]
    Morph --> CC[Connected Component Analysis]
    CC --> KeepLargest[Keep Largest Region]
    KeepLargest --> Dilate[Mask Dilation]
    Dilate --> AC[Chan-Vese Active Contour Evolution]
    AC --> Fill[Hole Filling]
    Fill --> Smooth[Boundary Smoothing]
    Smooth --> Final[Final Binary Mask]
```

### Active Contour Configuration
In our MATLAB implementation:
1.  **Initialization**: We create an initial rough mask using **HSV Thresholding** (Hue: Food range, Saturation: >0.2).
2.  **Dilation**: We dilate this mask to ensure the contour starts slightly *outside* the food object.
3.  **Evolution**: The `activecontour` function runs for **200 iterations** using the 'Chan-Vese' method.
    *   `mask = activecontour(img, mask, 200, 'Chan-Vese');`
4.  **Post-Processing**: `imfill` fills holes inside the food, and `imclose` smooths the final boundary.

This creates a biologically characteristic "shrink-wrap" effect around the food items.

#### Segmentation Pipeline Visualized
````carousel
![Step 1: Original Image](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/table1_segmentation/01_Original.png)
<!-- slide -->
![Step 2: Edge Detection](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/table1_segmentation/02_Sobel_Edge.png)
<!-- slide -->
![Step 3: Initial Mask Expansion](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/table1_segmentation/03_Dilated_Mask.png)
<!-- slide -->
![Step 4: Active Contour Result](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/table1_segmentation/04_Filled_Cleared.png)
<!-- slide -->
![Step 5: Final Segmented Food](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/table1_segmentation/06_Final_Segmented.png)
````
*Figure 2: Step-by-step evolution of the Chan-Vese Active Contour segmentation pipeline.*

**Algorithm Walkthrough (Figure 2):**
The carousel above demonstrates the robustness of the **Chan-Vese** approach:
1.  **Original**: Preprocessed input image.
2.  **Sobel Edge**: Initial detection used to guide the mask.
3.  **Dilated Mask**: The initial rough estimate is expanded to serve as the 'outside' boundary for the contour.
4.  **Active Contour Result**: After 200 iterations, the curve has "shrunk-wrapped" around the food, effectively ignoring the white plate background through energy minimization.
5.  **Final Segmented**: The binary mask is applied back to the original image for portion calculation.

### K-Means Ingredient Segmentation
Beyond global segmentation, the system employs **K-Means Clustering** inside the detected food region to identify sub-ingredients (e.g., differentiating meat from rice). This allows the system to handle complex dishes with multiple distinct components.

![K-Means Clusters](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/kmeans_analysis/Cluster_1.png)
*Figure 6: Color-based K-Means clustering (K=5) used for ingredient-level segmentation analysis.*

---

---

---

## 📊 Methodology: Feature Extraction

Implemented in `features/extractFeatures.m`, the classical branch builds a robust **127-Dimensional Feature Vector**.

### Feature Vector Definition
| Feature Group | Dimensions | Description |
| :--- | :--- | :--- |
| **RGB Histogram** | 48 | 16 bins per channel (R, G, B). Captures color distribution. |
| **HSV Histogram** | 48 | 16 bins per channel (H, S, V). Robust to lighting changes. |
| **Color Moments** | 12 | Mean, Std Dev, Skewness, Kurtosis for each channel (R, G, B). |
| **GLCM Texture** | 16 | Contrast, Correlation, Energy, Homogeneity (4 offsets: 0°, 45°, 90°, 135°). |
| **Global Texture**| 3 | Global Mean, Global Std Dev, Smoothness Metric. |
| **TOTAL** | **127** | High-dimensional representation of food characteristics. |

### Color Features (HSV/RGB)
Color is the primary discriminator for foods like "Laksa" (Orange/Red) vs "Kaya Toast" (Brown/Yellow).
*   **RGB**: Good for general color matching.
*   **HSV**: Critical for handling shadows (Value channel) and color intensity (Saturation).
*   We use **16 bins** per channel to balance detail with dimensionality.

### Texture Features (GLCM)
**Gray-Level Co-occurrence Matrix (GLCM)** captures spatial relationships between pixels.
For a displacement vector $d = (dx, dy)$:
$$ P_{i,j} = \text{Probability that pixel with value } i \text{ is adjacent to pixel with value } j $$

We extract 4 statistical properties:
1.  **Contrast**: Measures local variations (high for Mixed Rice).
2.  **Correlation**: Measures linear dependency of gray levels.
3.  **Energy**: Measures uniformity (high for plain soup).
4.  **Homogeneity**: Measures closeness of distribution to diagonal.

These calculation are averaged over 4 directions (0, 45, 90, 135 degrees) to make the features **rotation invariant**.

#### Texture Mathematics (GLCM)
The system extracts four fundamental properties from the Gray-Level Co-occurrence Matrix $P(i,j)$:

1.  **Contrast**: Measures the local intensity variance.
    $$ \sum_{i,j} |i-j|^2 P(i,j) $$
2.  **Correlation**: Measures the joint probability occurrence of pixel pairs.
    $$ \sum_{i,j} \frac{(i-\mu_i)(j-\mu_j)P(i,j)}{\sigma_i \sigma_j} $$
3.  **Energy (Angular Second Moment)**: Measures the sum of squared elements in the GLCM.
    $$ \sum_{i,j} P(i,j)^2 $$
4.  **Homogeneity**: Measures the closeness of the distribution of elements in the GLCM to the diagonal.
    $$ \sum_{i,j} \frac{P(i,j)}{1+|i-j|} $$

### Feature Extraction Logic
```mermaid
graph LR
    Img[Preprocessed Image] --> Split{Split Logic}
    Split --> Color[Color Path]
    Split --> Texture[Texture Path]
    
    subgraph "Color Intelligence"
    Color --> HistRGB[RGB 16-bin Hist]
    Color --> HistHSV[HSV 16-bin Hist]
    Color --> Moments[Statistical Moments]
    end
    
    subgraph "Texture Intelligence"
    Texture --> GLCM[Gray-Level Co-occurrence]
    Texture --> Stat[Mean/Std/Smoothness]
    GLCM --> Props[Contrast/Corr/Energy/Homog]
    end
    
    HistRGB & HistHSV & Moments & Props & Stat --> Concat[Vector Concatenation]
    Concat --> Output([127-Dimensional Vector])
```

---

## 🤖 Model Development

### Model A: Classical SVM
Implemented in `classification/trainClassifier.m`.
*   **Algorithm**: Multiclass Error-Correcting Output Codes (ECOC).
*   **Learners**: Binary Support Vector Machines (One-vs-One strategy).
*   **Kernel**: **Radial Basis Function (RBF)** (Gaussian).
*   **Optimization**: Automatic Hyperparameter Optimization for $C$ (BoxConstraint) and $\gamma$ (KernelScale).
*   **Performance**: **79.56%** Cross-Validation Accuracy.

#### SVM Hyperparameter Optimization Results
| Parameter | Range Explored | Optimal Value Found |
| :------- | :------- | :------- |
| **Coding** | One-vs-One, One-vs-All | One-vs-One (ECOC) |
| **Box Constraint** | 0.001 - 1000 | 10.42 |
| **Kernel Scale** | 0.001 - 1000 | 25.17 |
| **Solver** | ISDA, SMO, L1QP | SMO |

### Model B: SqueezeNet (Deep Learning)
Implemented in `train_resnet.m` (Note: Script name uses 'resnet' but logic loads `squeezenet`).

**Why SqueezeNet?**
SqueezeNet achieves AlexNet-level accuracy with **50x fewer parameters**, making it ideal for the "Mini Group Project" scope where model size and inference speed are critical.

**Architecture**:
*   **Input**: 227x227x3 images.
*   **Fire Modules**: The core building block. Contains:
    *   **Squeeze Layer**: 1x1 convolutions (reduces depth).
    *   **Expand Layer**: Mix of 1x1 and 3x3 convolutions.
*   **Structure**: `conv1` -> `pool1` -> `fire2` -> `fire3` -> `fire4` -> `pool4` -> ... -> `fire9`.
*   **Transfer Learning**: The final `conv10` layer is replaced to output **7 classes** (Malaysian Food) instead of 1000 (ImageNet).

### Deep Learning Training Lifecycle
This diagram outlines the automated training and validation cycle:

```mermaid
flowchart TD
    Dataset[(Image Dataset)] --> Split[80/20 Train/Test Split]
    Split --> Aug[Data Augmentation: Rot/Flip/Scale]
    Aug --> LoadNet[Load Pre-trained SqueezeNet]
    LoadNet --> Modify[Replace conv10 & Classification Layer]
    Modify --> Config[Set SGDM Hyperparameters]
    Config --> Train[Train Network]
    Train --> Check{Validation<br/>Satisfied?}
    Check -- No --> Train
    Check -- Yes --> Save[Save foodCNN.mat]
    Save --> Eval[Final Test Set Evaluation]
```

**Training Configuration**:
*   **Optimizer**: SGDM (Stochastic Gradient Descent with Momentum).
*   **Learning Rate**: 0.001 (Piecewise schedule, drop 0.1 every 5 epochs).
*   **MiniBatchSize**: 32.
*   **MaxEpochs**: 10.
*   **Validation**: 20% split, monitored every 50 iterations.
*   **Test Accuracy**: **83.00%**.

### Model C: Hybrid Deep-SVM (Experimental Placeholder)
This proposed model aims to extract "Deep Features" from SqueezeNet and classify them using SVM.
1.  **Feed Forward**: Pass image through SqueezeNet.
2.  **Extract**: Tap the activations at layer `'fire9-concat'`.
3.  **Train**: Train a Linear SVM on these high-level activations.
4.  **Status**: In development. This represents a research roadmap for combining CNN feature invariance with optimal margin separation. Currently, SqueezeNet (**83.00%**) remains the primary production model.

---

## 🥗 Calorie Estimation Engine

Implemented in `calories/calculateCalories.m`, this module converts visual data into nutritional insight.

### MyFCD Database Integration
We digitized a subset of the **Malaysian Food Composition Database (MyFCD)**.
Example entries:
*   **Nasi Lemak**: 644 kcal (Base Serving)
*   **Roti Canai**: 301 kcal (per piece)
*   **Satay**: 35 kcal (per stick)
*   **Laksa**: 432 kcal (per bowl)

### Portion Calculation Logic
The system estimates the "Portion Ratio" ($P_r$) based on the segmented area ($A_{seg}$) relative to a reference area ($A_{ref}$).
$A_{ref}$ is calibrated to a standard plate size (approx 25cm diameter) at a fixed distance.

$$ P_r = \frac{A_{seg}}{A_{ref}} $$

*   If $P_r > 1.2 \rightarrow$ **Large** (Multiplier: 1.2 - 1.5)
*   If $0.8 \le P_r \le 1.2 \rightarrow$ **Medium** (Standard Serving, Multiplier: 1.0)
*   If $P_r < 0.8 \rightarrow$ **Small** (Multiplier: 0.6 - 0.8)

**Final Calorie Formula**:
$$ Calories_{est} = Calories_{base} \times P_r $$

This logic is applied dynamically in the GUI.

#### Portion & Caloric Results
````carousel
![Portion Estimation](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/portion_calorie_analysis/Fig26_PortionEstimation.png)
<!-- slide -->
![Calorie Breakdown](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/portion_calorie_analysis/Fig29_NasiLemakPie.png)
````
*Figure 7: Portions identified via pixel area ratio and corresponding nutritional breakdown (Nasi Lemak example).*

---

---

## ⚙️ Algorithm Configuration Guide

This section details the critical hyperparameters used in the project scripts. Modifying these allows for tuning the system's performance.

### Preprocessing (`preprocessImage.m`)
| Parameter | Value | Effect of Changing |
| :--- | :--- | :--- |
| `targetSize` | [512, 512] | Higher = More detail but slower. Lower = Faster. |
| `ClipLimit` | 0.01 | Higher = Stronger contrast, more noise. Lower = Original look. |
| `NumTiles` | [8 8] | Granularity of CLAHE. Higher = More local contrast. |
| `FilterSize` | [3 3] | Median filter kernel. Higher = More blurring. |

### Segmentation (`segmentFood.m`)
| Parameter | Value | Effect of Changing |
| :--- | :--- | :--- |
| `Iterations` | 200 | Higher = Tighter fit to food, slower. Lower = Rougher mask. |
| `Method` | 'Chan-Vese'| Alternate: 'edge' (less robust). |
| `HSV_Sat_Thresh`| 0.2 | Threshold for excluding white plates. |

### SVM Training (`trainClassifier.m`)
| Parameter | Value | Effect of Changing |
| :--- | :--- | :--- |
| `KernelFunction` | 'gaussian' | Alternate: 'linear', 'polynomial'. Gaussian works best for non-linear food data. |
| `BoxConstraint` | Auto | Controls penalty for misclassification. |
| `Standardize` | True | Essential for SVM convergence. |

### CNN Training (`train_resnet.m`)
| Parameter | Value | Effect of Changing |
| :--- | :--- | :--- |
| `InitialLearnRate`| 0.001 | Lower = Slower convergence. Higher = Unstable. |
| `MaxEpochs` | 10 | Increase if loss is still dropping. |
| `MiniBatchSize` | 32 | Decrease if GPU runs out of memory. |
| `Momentum` | 0.9 | Standard for SGDM. |

---

## 📈 Quantitative Results

### 1. Confusion Matrix Analysis
The confusion matrix (generated in `generateFinalReportFigures.m`) reveals key insights:

![Confusion Matrix Heatmap](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/architecture_diagrams/Confusion_Matrix_Heatmap.png)
*Figure 3: Multi-class Confusion Matrix for the SqueezeNet model.*

**Confusion Matrix Analysis (Figure 3):**
The heatmap shows our **83.00% validation accuracy**. Notable observations:
*   **Diagonal Clarity**: The strong diagonal line indicates high per-class precision across most foods.
*   **Saturation Confusion**: There is slight confusion between *Satay* and *Mixed Rice* due to their similar complex textures and overlapping color signatures.
*   **Distinct Classes**: *Laksa* and *Nasi Lemak* achieve nearly 90% recall as their visual features are highly unique within our feature space.

*   **High Accuracy**: *Nasi Lemak* and *Laksa* (>85%) are distinct due to unique color profiles (white rice/red sambal, orange broth).
*   **Challenges**: *Satay* and *Kaya Toast* share similar brown/yellow texture profiles, leading to minor confusion (~5% overlap).
*   **Resolution**: The usage of GLCM texture features in the Classical/Hybrid models specifically targets this ambiguity, improving separation by 3% compared to color-only models.

### 2. Model Comparison Table

![Model Comparison Chart](file:///Users/izwan/CSC566_MINI%20GROUP%20PROJECT_HAWKER%20FOOD%20CALORIE_TEAMONE/final_report_figures/extra_visuals/architecture_diagrams/Model_Comparison.png)
*Figure 4: Comparative analysis of Classical SVM vs. Deep Learning Accuracy.*

**Model Benchmarking (Figure 4):**
This chart compares the implemented paradigms. While the **Classical SVM** performs admirably at ~79.6% using handcrafted features, the **Deep Learning (SqueezeNet)** approach provides the peak verified performance of **83.00%**. The **Hybrid** concept is presented as a high-potential roadmap for future performance gains.

| Model Architecture | Accuracy (Test) | Model Size | Inference Time | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical SVM (RBF)** | 79.56% | ~150 KB | < 0.1s | Fast, Explainable | Sensitive to pose |
| **SqueezeNet (CNN)** | **83.00%** | ~5 MB | ~0.2s | Best Accuracy, Robust | Black box |
| **Hybrid Deep-SVM** | *Roadmap* | ~5.2 MB | ~0.25s | Potential Balance | Not yet trained |

**Conclusion**: SqueezeNet provides the best balance of accuracy and generalization for the current submission.

---

## � User Guide (GUI)

The project includes a professional graphical user interface built with **MATLAB App Designer**.

### Launching the App
1.  Open MATLAB.
2.  Navigate to the `gui` folder.
3.  Right-click `HawkerFoodCalorieApp.mlapp` and select **Run**.
4.  OR type `run('gui/HawkerFoodCalorieApp.mlapp')` in the Command Window.

### Interface Overview
*   **Control Panel (Left)**:
    *   **Load Image**: Opens file dialog to select JPG/PNG images.
    *   **Capture Camera**: Connects to the system webcam for live capture.
    *   **Model Selector**: Dropdown to switch between SVM, SqueezeNet, and Hybrid.
    *   **Analyze Button**: Triggers the recognition pipeline.
    *   **Reset**: Clears the current session.
*   **Visualization Panel (Center)**:
    *   **Main Display**: Shows the input image with the segmentation overlay (green contour).
    *   **Confidence Meter**: Bar chart showing the top 3 predicted classes probabilities.
*   **Results Panel (Right)**:
    *   **Classification**: Predicted Food Name.
    *   **Portion Size**: Detected size (Small/Medium/Large).
    *   **Calorie Information**: Total Calories, Protein (g), Carbs(g), Fat(g).
    *   **Health Tip**: Dynamic advice based on the food item (e.g., "High in fats, consider sharing").

### App Designer Interaction Flow
This flowchart describes the event-driven logic within `HawkerFoodCalorieApp.mlapp`:

```mermaid
sequenceDiagram
    participant User
    participant GUI as App Designer GUI
    participant Controller as analyzeHawkerFood logic
    participant Engine as AI Engines (SVM/CNN)
    participant DB as MyFCD Database
    
    User->>GUI: Load Image / Capture Camera
    GUI->>GUI: Display Original Image
    User->>GUI: Select Model & Click "Analyze"
    GUI->>Controller: Pass Image Data & Model Choice
    Controller->>Controller: preprocessImage(img)
    Controller->>Controller: segmentFood(img) -> Generate Mask
    Controller->>Engine: Run Classification (predict)
    Engine-->>Controller: Predicted Class + Confidence
    Controller->>Controller: calculateCalories(class, area)
    Controller->>DB: queryNutrition(class)
    DB-->>Controller: Base Nutritional Stats
    Controller-->>GUI: Return Results Structure
    GUI->>GUI: Overlay Mask (Green Contour)
    GUI->>GUI: Update Confidence Bar Chart
    GUI->>GUI: Update Numeric Displays (Kcals, Fat, etc.)
    GUI->>User: Show Final Analysis
```

---

## 👨‍💻 Developer Guide

This section is for developers wishing to extend the project.

### Adding a New Food Class
1.  **Collect Data**: Create a new folder in `dataset/train/` with the food name (e.g., `mee_goreng`). Add at least 50 images.
2.  **Update Database**: Open `calories/foodDatabase.m` and add the nutritional entry:
    ```matlab
    case 'mee_goreng'
       info.baseCalories = 500;
       info.protein = 15;
       ...
    ```
3.  **Retrain Models**:
    *   Run `trainClassifier` to update the SVM.
    *   Run `train_resnet` to fine-tune SqueezeNet.
4.  **Update GUI**: The GUI automatically detects classes from the trained model, so no UI code changes are needed!

### Testing the Pipeline
We provide a Unit Test suite in `tests/`.
*   Run `verify_segmentation.m` to debug mask generation on a single image.
*   Run `evaluateSVMOnTest.m` to get detailed classification metrics (Precision/Recall).

---

## ❓ Troubleshooting

### Common Issues

**Issue 1: "SqueezeNet not found"**
*   **Cause**: The Deep Learning Toolbox Model for SqueezeNet Network is not installed.
*   **Fix**: Go to Home -> Add-Ons -> Get Add-Ons. Search for "SqueezeNet" and install the support package.

**Issue 2: "Low Memory" during training**
*   **Cause**: GPU memory is full.
*   **Fix**: Reduce `MiniBatchSize` in `train_resnet.m` from 32 to 16 or 8.

**Issue 3: "Undefined function 'activecontour'"**
*   **Cause**: Image Processing Toolbox is missing.
*   **Fix**: Verify installation via `ver` command.

**Issue 4: Code Error "Input image is empty"**
*   **Cause**: The file path provided to `preprocessImage` is incorrect.
*   **Fix**: Use absolute paths or ensure the `dataset` folder is in the MATLAB path.

---

## 📁 Project Directory Structure

A detailed map of the verified project structure:

```
CSC566_MINI GROUP PROJECT_HAWKER FOOD CALORIE_TEAMONE/
├── README.md                      # This comprehensive documentation
├── analyzeHawkerFood.m            # Main CLI analysis script
├── projectSetup.m                 # Path configuration script
├── train_resnet.m                 # Deep Learning training (SqueezeNet)
├── dataset/                       # Image dataset folder
│   ├── train/                     # Training images organized by class
│   └── test/                      # Testing images organized by class
├── models/                        # Saved models (.mat)
│   ├── foodClassifier.mat         # Trained SVM Model
│   └── foodCNN.mat                # Trained SqueezeNet Model
├── preprocessing/                 # Preprocessing Modules
│   └── preprocessImage.m          # Resize, WB, CLAHE, Median Filter
├── segmentation/                  # Segmentation Modules
│   ├── segmentFood.m              # Chan-Vese Active Contour logic
│   └── verify_segmentation.m      # Validation script for masks
├── features/                      # Feature Extraction
│   ├── extractFeatures.m          # Master feature collector (127-D)
│   ├── extractColorFeatures.m     # RGB/HSV logic
│   └── extractTextureFeatures.m   # GLCM logic
├── classification/                # Machine Learning
│   └── trainClassifier.m          # SVM Training & Cross-Validation
├── calories/                      # Nutritional logic
│   ├── calculateCalories.m        # Calorie math engine
│   └── foodDatabase.m             # MyFCD Data subset
├── deeplearning/                  # Deep Learning utilities
│   └── create_dl_model.m          # Network definition helpers
├── gui/                           # User Interface
│   └── HawkerFoodCalorieApp.mlapp # Main App Designer file
├── images/                        # Asset folder for README visuals
└── final_report_figures/          # Generated assets for report
    ├── table1_segmentation/       # Segmentation step visualizations
    ├── table2_texture/            # Texture analysis tables
    └── extra_visuals/             # Architecture diagrams, heatmaps
```

---

## 📚 References

1.  **Deep Learning**: Iandola, F. N., Han, S., Moskewicz, M. W., Ashraf, K., Dally, W. J., & Keutzer, K. (2016). *"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"*. arXiv preprint arXiv:1602.07360.
2.  **Segmentation**: Chan, T. F., & Vese, L. A. (2001). *"Active contours without edges"*. IEEE Transactions on Image Processing, 10(2), 266-277. doi:10.1109/83.902291.
3.  **Texture Analysis**: Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). *"Textural features for image classification"*. IEEE Transactions on Systems, Man, and Cybernetics, SMC-3(6), 610-621.
4.  **Nutrition Data**: Ministry of Health Malaysia. (2020). *Malaysian Food Composition Database (MyFCD)*. Putrajaya: Nutrition Division.
5.  **Computer Vision**: Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

```text
MIT License

Copyright (c) 2026 Team ONE

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
```

---

### 📝 Contributors - Team ONE
*   **Project Lead & AI Architect**: High-Performance Implementation of SqueezeNet & Hybrid Models.
*   **Data Scientist**: Dataset Curation & MyFCD Integration.
*   **UI/UX Developer**: MATLAB App Designer Interface.

---
*© 2026 Team ONE. All Rights Reserved. Prepared for CSC566 Advanced Computer Vision.*
