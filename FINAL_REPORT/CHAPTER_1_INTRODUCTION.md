# CHAPTER 1: INTRODUCTION

This chapter provides a comprehensive overview of the **Malaysian Hawker Food Calorie Estimation System**. It outlines the context of health issues driving the project, defines the specific problems addressed, and details the objectives, scope, and significance of the work.

## 1.1 Background of the Study

Obesity and diet-related non-communicable diseases have emerged as critical public health concerns in Malaysia. The high prevalence of these conditions is frequently attributed to the consumption of calorie-dense local dishes, which are readily available at hawker centers. Detailed dietary monitoring is essential for managing personal health but remains a challenging task for the general public due to the difficulty in accurately estimating portion sizes and caloric content of diverse food items. Traditional methods of manual calorie counting are often tedious and prone to significant estimation errors, which leads to poor adherence to dietary goals. This necessitates a technological solution that can automate the process of food recognition and nutritional assessment.

To address this issue, this project presents an intelligent vision-based system capable of recognizing Malaysian hawker foods and ensuring accurate calorie estimation. The development of accessible tools for nutrition monitoring addresses challenges like diabetes by utilizing image processing to enable automated recognition and calorie estimation from food images. By integrating advanced image processing algorithms with machine learning models, the system seeks to provide users with immediate and reliable nutritional information. This approach not only simplifies the tracking of daily caloric intake but also empowers individuals to make informed dietary choices.

## 1.2 Problem Statement

The absence of integrated systems that combine cultural specificity with precise portion analysis limits the effectiveness of health interventions in diverse populations. By focusing on hawker foods, which are affordable and ubiquitous yet nutritionally variable, the project bridges this divide, offering a tool that empowers users to make informed choices without sacrificing culinary traditions. The lack of such systems also overlooks the potential for technology to support tourism and education, where visitors could learn about local foods while monitoring intake.

## 1.3 Objectives

The following are the objectives of this project:

1.  To develop a food recognition system using classical image processing techniques for pre-processing and segmentation, with dual classification options using either Support Vector Machine (SVM) or Convolutional Neural Network (CNN).
2.  To implement a graphical user interface prototype that integrates the system for immediate dietary feedback, supporting public health initiatives in Malaysia.
3.  To evaluate the system's performance on a subset of the Malaysia Food 11 dataset comprising 7 Malaysian hawker food classes, and measure calorie estimation accuracy.

## 1.4 Scope of the Project

The project focuses on recognizing and analyzing seven key Malaysian hawker food classes from the Malaysia Food 11 dataset: **Nasi Lemak**, **Roti Canai**, **Satay**, **Laksa**, **Popiah**, **Kaya Toast**, and **Mixed Rice**. It encompasses image pre-processing, classification, segmentation, portion estimation, and calorie calculation using MATLAB tools. Classical methods include histogram enhancement, noise filtering, HSV thresholding, morphological operations, k-means clustering, region descriptors, GLCM texture analysis, and Chan-Vese active contours.

The system provides dual classification options using either Support Vector Machine (SVM) or Convolutional Neural Network (CNN) architecture (specifically **SqueezeNet**). The system processes single-plate RGB images from real-world scenarios, excluding multi-plate or non-food images. Calorie estimates rely on the Malaysian Food Composition Database (MyFCD) for base values, adjusted by portion ratios using shape compactness and color density analysis. A GUI has been developed for demonstration, but deployment to mobile apps is beyond the scope. Testing involves accuracy metrics on dataset subsets. Limitations include handling only static images, not videos, and assuming standard plate compositions without extreme occlusions or unusual presentations.

## 1.5 Significance of Project

This project holds substantial value for public health and cultural preservation in Malaysia, where hawker food represents a UNESCO-listed intangible heritage yet contributes to rising obesity rates. By automating recognition and calorie estimation, it empowers individuals to make informed dietary choices, aligning with national initiatives like the Malaysian Healthy Plate program. The dual-mode approach innovates by combining classical image processing with an option for deep learning classification, improving accessibility in resource-limited settings.

Similar systems have shown promise in dietary interventions, as seen in Central Asian contexts where datasets enable personalized nutrition, as demonstrated by **Karabay et al. (2023)**. Moreover, recent work on large-scale food scene datasets by **Karabay et al. (2025)** demonstrates that region-specific food recognition systems are feasible and accurate even in complex real-world settings. Economically, it supports tourism by aiding visitors in identifying local foods while promoting balanced consumption. Academically, it advances image processing applications, demonstrating how segmentation enhances calorie accuracy in diverse cuisines, as shown in the research of **Haque et al. (2022)**. Ultimately, the prototype could integrate into health apps, fostering preventive care and reducing healthcare burdens associated with diet-related diseases. Beyond immediate impacts, it encourages further research into culturally adapted technologies, potentially inspiring similar systems for other regional cuisines and contributing to global efforts in combating non-communicable diseases through innovative digital solutions.
