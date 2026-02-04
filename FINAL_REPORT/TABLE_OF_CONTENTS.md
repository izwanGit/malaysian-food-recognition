# FINAL REPORT: MALAYSIAN HAWKER FOOD CALORIE ESTIMATION SYSTEM

## TABLE OF CONTENTS

---

### CHAPTER 1: INTRODUCTION
1.1 Background of the Study
1.2 Problem Statement
1.3 Objectives
1.4 Scope of the Project
1.5 Significance of Project

---

### CHAPTER 2: LITERATURE REVIEW
2.1 Domain Area: Health and Nutrition
2.2 Research Area: Image Processing Techniques
2.3 Research Area: Machine Learning Approaches
2.4 Summary of Reviewed Works

---

### CHAPTER 3: METHODOLOGY
3.1 Overall System Flow
3.2 Dataset / Image Source
3.3 Image Processing Techniques Used
   - 3.3.1 Pre-processing and Data Augmentation
   - 3.3.2 Feature Extraction: Color Histograms and GLCM Texture
   - 3.3.3 Classification: SVM and CNN
   - 3.3.4 Segmentation: HSV Thresholding, Morphology, and Chan-Vese
   - 3.3.5 Portion Estimation and Calorie Calculation
   - 3.3.6 Summary of Image Processing Techniques
3.4 Tools and Software
3.5 Hardware Requirements
3.6 Conclusion

---


### CHAPTER 4: IMPLEMENTATION
4.1 Algorithm Design
   - 4.1.1 SVM Pipeline (analyzeHawkerFood.m)
   - 4.1.2 CNN Pipeline (analyzeHawkerFoodDL.m)
4.2 Code Structure
   - 4.2.1 GUI Module (HawkerFoodCalorieApp.m)
   - 4.2.2 Training Scripts (trainClassifier.m, trainCNN.m)
   - 4.2.3 Helper Functions (preprocessing, features, segmentation, calories)
4.3 Parameter Settings
   - 4.3.1 Pre-processing Parameters
   - 4.3.2 Feature Extraction Parameters
   - 4.3.3 Segmentation Parameters
   - 4.3.4 SVM Classifier Parameters
   - 4.3.5 CNN Training Parameters
   - 4.3.6 Calorie Estimation Parameters

---

### CHAPTER 5: RESULTS AND DISCUSSION
5.1 Experimental Results
   - 5.1.1 Classification Results: SVM vs CNN
   - 5.1.2 Sample Input and Output Results: Segmentation
5.2 System Prototype: GUI Walkthrough
   - 5.2.1 System Interaction Workflow
   - 5.2.2 Complete Analysis Pipeline
   - 5.2.3 Portion and Calorie Estimation Logic
   - 5.2.4 Classification Mode Comparison and Multi-Class Demonstration
5.3 Performance Evaluation
   - 5.3.1 Accuracy, Precision, Recall, F1-Score
   - 5.3.2 Confusion Matrix Analysis
5.4 Discussion

---

### CHAPTER 6: CONCLUSION AND FUTURE WORK
6.1 Conclusion
6.2 Future Improvements

---

### REFERENCES
