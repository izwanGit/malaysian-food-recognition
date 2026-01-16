# Malaysian Hawker Food Recognition with Portion-Based Calorie Estimation

**CSC566 Mini Group Project - Team One**

## Team Members
- Muhammad Izwan bin Ahmad (2024938885)
- Ahmad Azfar Hakimi bin Mohammad Fauzy (2024544727)
- Afiq Danial bin Mohd Asrinnihar (2024974673)
- Alimi bin Ruzi (2024568765)

---

## Quick Start

### 1. Setup Project
```matlab
cd '/Applications/MATLAB_R2025a.app/toolbox/images/imdata/CSC566_MINI GROUP PROJECT_HAWKER FOOD CALORIE_TEAMONE'
projectSetup()
```

### 2. Train Classifier (First Time Only)
```matlab
trainClassifier()  % Takes 5-10 minutes
```

### 3. Run Demo
```matlab
demo()
```

### 4. Open GUI
```matlab
HawkerFoodCalorieApp()
```

---

## Project Structure

```
├── projectSetup.m          % Project initialization
├── analyzeHawkerFood.m     % Main analysis pipeline
├── displayResults.m        % Results visualization
├── demo.m                  % Quick demo script
│
├── preprocessing/          % Image pre-processing
├── features/               % Feature extraction
├── classification/         % Food classification
├── segmentation/           % Food segmentation
├── portion/                % Portion estimation
├── calories/               % Calorie calculation
├── gui/                    % GUI application
├── tests/                  % Test suite
├── models/                 % Trained models
└── dataset/train/          % Training images (linked)
```

---

## Supported Food Classes

| Food | Base Calories |
|------|---------------|
| Nasi Lemak | 650 kcal |
| Roti Canai | 300 kcal |
| Satay | 200 kcal |
| Laksa | 500 kcal |
| Popiah | 185 kcal |
| Kaya Toast | 300 kcal |
| Mixed Rice | 620 kcal |

---

## Usage Example

```matlab
% Analyze a single image
results = analyzeHawkerFood('path/to/food_image.jpg');

% Display results
disp(results.foodClass);      % e.g., 'nasi_lemak'
disp(results.calories);       % e.g., 650
disp(results.portionLabel);   % e.g., 'Medium'
```

---

## Run Tests

```matlab
run('tests/testPreprocessing.m')
run('tests/testFeatureExtraction.m')
run('tests/testSegmentation.m')
run('tests/testFullPipeline.m')
```

---

## Requirements
- MATLAB R2020a or later
- Image Processing Toolbox
- Statistics and Machine Learning Toolbox
