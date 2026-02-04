# CHAPTER 2
 
## LITERATURE REVIEW
 
### 2.1 Introduction
 
This chapter reviews the theoretical background and related works necessary for the development of the Malaysian Hawker Food Recognition and Calorie Estimation System. The review focuses on three primary pillars as illustrated in the project’s conceptual map: the domain area of health and nutrition, the research area of image processing techniques, and the research area of machine learning approaches. Figure 2.1 presents the conceptual map of the proposed system, outlining the flow from domain understanding to technical implementation. The domain area establishes the necessity of dietary assessment and the utilization of nutritional databases such as MyFCD. The research area is further divided into image processing, which covers segmentation and feature extraction, and machine learning, which compares traditional methods like Support Vector Machine (SVM) with modern Deep Learning (CNN) architectures.
 
**Figure 2.1: Conceptual Map of the Malaysian Hawker Food Recognition and Calorie Estimation System**
 
![Conceptual Map](../final_report_figures/conceptual_map.png)
 
### 2.2 Domain Area: Health and Nutrition
 
According to Niu et al. (2024), the rising prevalence of diet-related chronic diseases, particularly obesity and diabetes mellitus, has become a critical public health concern in many countries including Malaysia. As noted by Mognard et al. (2024), over the past two decades, rapid urbanisation, changing lifestyles, and increased consumption of high-calorie foods including hawker foods have contributed significantly to this trend. Recent national surveys indicate that overweight and obesity rates continue to climb across all age groups, with a notable acceleration among young adults and working populations who frequently rely on hawker centre meals.
 
Effective weight management and diabetes prevention require a precise balance between energy intake and energy expenditure. Based on the systematic review by Dalakleidi et al. (2022), traditional dietary assessment methods, such as 24-hour dietary recalls and food frequency questionnaires, have long been the standard tools in nutritional epidemiology. However, these methods are prone to substantial human error, including recall bias, underreporting of portion sizes, and social desirability bias. According to Tahir and Loo (2021), studies consistently show that such conventional approaches can underestimate actual caloric intake by 20 to 30 percent, especially among populations with low nutritional literacy. As highlighted by Niu et al. (2024), consequently, there is a growing demand for automated, technology-driven systems capable of estimating caloric intake directly from food images, thereby reducing the cognitive burden on users and improving compliance in long-term dietary monitoring.
 
A reliable nutritional database forms the foundation of any credible calorie estimation system. In the Malaysian context, the Malaysian Food Composition Database (MyFCD), maintained by the Ministry of Health Malaysia, serves as the primary reference. First established in 1997, MyFCD has undergone significant updates, with the most comprehensive revision released in 2017. This version expanded coverage to include a wider variety of raw ingredients, processed foods, and commonly prepared dishes, including many traditional hawker items and kuih. Recent work has further highlighted the need to enrich MyFCD with detailed sugar profiling. Norizan et al. (2025) analysed the sugar content of numerous Malaysian foods and beverages, revealing high sucrose levels in popular items such as kuih, cereals, and sweetened drinks. Their findings provide essential data for updating MyFCD and enable more accurate estimation of free sugar intake among the population.
 
According to Mognard et al. (2024), the integration of datasets like MyFCD into an automated recognition system allows recognised food items to be mapped directly to their corresponding macronutrient and energy values. For example, once the system identifies a dish, it can retrieve the caloric density per standard serving size from MyFCD and adjust the estimation according to the detected portion size. This linkage between visual recognition and nutritional quantification is central to the proposed system. However, the overall accuracy of calorie estimation depends heavily on two upstream processes: correct food classification and reliable portion size or volume estimation. These technical challenges fall squarely within the domains of image processing and machine learning, which are examined in the following sections.
 
### 2.3 Research Area: Image Processing Techniques
 
Image processing constitutes the essential preprocessing and feature preparation stage in any food recognition pipeline. At its core, a digital image is represented as a two-dimensional array of pixels, where each pixel contains intensity values. Grayscale images use single intensity values, while colour images typically employ multi-channel representations such as Red-Green-Blue (RGB) or Hue-Saturation-Value (HSV) colour spaces. Before classification can occur, raw images captured by mobile devices must be transformed into a form that highlights relevant visual information while suppressing noise and irrelevant background elements.
 
The first step is image preprocessing. As explained by Niu et al. (2024), food photographs taken in various settings such as hawker centres typically suffer from inconsistent lighting, varying angles, different resolutions, and cluttered backgrounds. According to Liu et al. (2023), standard preprocessing operations include resizing all images to a uniform dimension, applying noise reduction filters, and performing colour space conversion when necessary. Enhancement techniques such as histogram equalisation are commonly applied to improve contrast and normalise intensity distributions across images. Gaussian filtering and median filtering are commonly employed to smooth images and reduce artefacts caused by poor illumination. Proper preprocessing ensures that subsequent stages receive consistent input, which directly improves the stability and accuracy of both segmentation and classification models.
 
Following preprocessing, image segmentation partitions the photograph into meaningful regions, isolating the food item from the plate, table, or surrounding environment. This step is particularly demanding for Malaysian hawker dishes, which are often served as mixed components on a single plate. Based on the review by Tahir and Loo (2021), techniques such as thresholding, K-means clustering, and background subtraction have been widely applied in early systems. Edge detection methods, including Canny and Sobel operators, help identify object boundaries by detecting rapid changes in pixel intensity. Morphological operations such as erosion, dilation, opening, and closing are frequently used to refine segmentation masks by removing noise and filling gaps. According to Dalakleidi et al. (2022), more recent approaches leverage deep learning-based segmentation models to achieve finer boundaries even in highly complex scenes. Accurate segmentation is crucial because any inclusion of background pixels or exclusion of food regions can lead to substantial errors in both recognition and subsequent volume estimation.
 
Once the region of interest is isolated, feature extraction converts the segmented image into a compact mathematical representation suitable for classification. Traditional handcrafted features remain relevant in many hybrid systems. As noted by Islam et al. (2024), colour features are commonly extracted using histograms or statistical moments in RGB or HSV colour spaces. According to Liu et al. (2023), texture features, frequently derived from the Gray-Level Co-occurrence Matrix (GLCM), help distinguish between foods that share similar colours but possess different surface patterns. Niu et al. (2024) highlight that shape descriptors and local binary patterns have also been utilised to capture geometric characteristics of individual food items.
 
Recent comprehensive reviews confirm that while handcrafted features offer interpretability and lower computational cost, they often fail to capture the high intra-class variability and inter-class similarity inherent in real-world food images. Tahir and Loo (2021) surveyed over a hundred studies and concluded that systems relying solely on handcrafted features generally achieve lower classification accuracy compared with deep learning approaches, particularly when dealing with culturally diverse and multi-component dishes. Figure 2.2 illustrates the general architecture of image-based food recognition systems as summarised by Dalakleidi et al. (2022), showing the complete pipeline from user input to calorie estimation.

**Figure 2.2: Architecture of Image-Based Food Recognition Systems (Adapted from Dalakleidi et al., 2022)**

![Architecture of Image-Based Food Recognition Systems](./figure_from_article/Figure%202:%20Architecture%20of%20Image-Based%20Food%20Recognition%20Systems.png)
 
### 2.4 Research Area: Machine Learning Approaches
 
The classification stage represents the core intelligence of the food recognition system. As summarized by Niu et al. (2024), two broad categories of machine learning techniques have been extensively studied: traditional algorithms that depend on handcrafted features and modern deep learning architectures that learn features directly from raw pixel data.
 
Support Vector Machine (SVM) is one of the most prominent traditional classifiers used in food recognition research. According to Islam et al. (2024), SVM seeks an optimal hyperplane that maximises the margin between different classes in the feature space. When combined with carefully engineered features such as colour histograms and GLCM texture descriptors, SVM has demonstrated respectable performance on relatively small datasets. Dalakleidi et al. (2022) note that its mathematical clarity and resistance to overfitting make it attractive for scenarios with limited training samples. Nevertheless, the effectiveness of SVM is strongly tied to the quality of the preceding feature extraction stage. If the handcrafted features do not adequately represent the subtle visual differences between similar hawker dishes, classification performance deteriorates rapidly. Table 2.1 summarises several image-based food recognition systems that employ handcrafted features with traditional classifiers on publicly available datasets.

**Table 2.1: Summary of Image-Based Food Recognition Systems Using Handcrafted Features and Traditional Classifiers (Adapted from Dalakleidi et al., 2022)**

| Segmentation | Feature Extraction | Classifier | Dataset | Accuracy |
|--------------|-------------------|------------|---------|----------|
| - | SIFT, LBP, Gabor, Color | SVM | NTU-FOOD | 62.7% |
| K-Means Clustering | SURF, Shape, Color | Borda Count | Ambient Kitchen | P=86.29%, R=83.61% |
| Superpixels | Mid-level Food Parts | SVM | UEC-Food100 | 60.50% |
| - | SIFT, SURF | SVM (BOF) | UEC-Food100 | 82.38% |
| - | Metric Forests | - | Food-101 | 68.29% |
| Fuzzy Clustering | Wavelet Kernel | Whale-LM ANN | UNIMIB2016 | 96.27% |
| Canny Edge, Multi-scale | Color, Texture, SIFT, SURF | 3-Layer ANN | UNIMIB2016 | 94.5% |
| Local Variation | Color, Texture, SIFT, MDSIFT | Multi-kernel SVM | UNIMIB2016 | 93.9% |
| Hough Transform | CEDD, Gabor, LBP | KNN | UNIMIB2015 | 99.05% |


In contrast, Convolutional Neural Networks (CNNs) have emerged as the dominant approach in contemporary food image recognition. Based on the comprehensive review by Liu et al. (2023), unlike traditional methods, CNNs automatically learn hierarchical feature representations through multiple layers of convolution, pooling, and non-linear activation. Early convolutional layers detect low-level features such as edges and textures, while deeper layers capture high-level semantic patterns that correspond to entire food items. As discussed by Tahir and Loo (2021), popular architectures including AlexNet, ResNet, VGG, and MobileNet have been adapted and fine-tuned for food recognition tasks with considerable success.
 
According to Niu et al. (2024), transfer learning has further accelerated progress in this field. By initialising a network with weights pre-trained on large general datasets such as ImageNet and then fine-tuning on domain-specific food images, researchers can achieve high accuracy even when labelled Malaysian food data is scarce. Dalakleidi et al. (2022) report that several recent studies focusing on various cuisines have reported classification accuracies exceeding 90 percent when employing transfer learning with modern CNN backbones.
 
Liu et al. (2023) emphasize that comparative analyses consistently show that CNN-based models outperform traditional machine learning approaches, especially when the dataset contains high visual variability. However, CNNs demand significantly larger training datasets and greater computational resources during both training and inference. This trade-off between accuracy and resource efficiency remains an important consideration when designing mobile-friendly dietary monitoring applications for the Malaysian population.
 
### 2.5 Summary of Reviewed Works
 
The literature reviewed in this chapter reveals a clear evolutionary trajectory in food recognition technology. According to Tahir and Loo (2021), early systems relied predominantly on conventional image processing pipelines coupled with traditional classifiers such as SVM. While these approaches provided acceptable performance on controlled datasets, they struggled with the complexity, intra-class variation, and multi-component nature of various foods including Malaysian hawker foods.
 
The advent of deep learning, particularly CNN architectures, has substantially elevated recognition accuracy and robustness. Comprehensive surveys such as the one conducted by Tahir and Loo (2021) confirm that deep neural networks now dominate state-of-the-art results in food image analysis. As pointed out by Liu et al. (2023), nevertheless, most high-performing models were trained on Western-centric datasets such as Food-101 or UEC-Food256. These datasets contain limited representation of Asian cuisines and virtually no coverage of authentic Malaysian hawker dishes such as Nasi Lemak, Char Kway Teow, or Roti Canai. Table 2.2 presents a comprehensive overview of publicly available food image datasets, illustrating the geographical and cultural bias towards American, Japanese, and European cuisines.

**Table 2.2: Publicly Available Food Image Datasets (Adapted from Tahir and Loo, 2021)**

| Year | Dataset | Food Category | Total Images/Classes | Image Source |
|------|---------|---------------|---------------------|---------------|
| 2009 | PFID | American Fast Foods | 1,038 (61) | Fast food restaurants |
| 2012 | UECFOOD-100 | Japanese Foods | 14,361 (100) | Mobile camera |
| 2012 | ChineseFoodNet | Chinese Dishes | 192,000 (208) | Web crawled |
| 2014 | Food-101 | American Foods | 101,000 (101) | Web crawled |
| 2014 | UECFOOD-256 | Japanese Foods | 25,088 (256) | Mobile camera |
| 2014 | UNICT-FD889 | Italian Foods | 3,583 (899) | Smartphone |
| 2016 | Vireo-Food 172 | Chinese Foods | 110,241 (172) | Web downloaded |
| 2017 | THFood-50 | Thai Foods | 700 (50) | Web downloaded |
| 2017 | Turkish-Foods-15 | Turkish Dishes | 7,500 (15) | Existing datasets |
| 2017 | Indian Food Database | Indian Foods | 5,000 (50) | Web downloaded |
| 2020 | Pakistani Food Dataset | Pakistani Dishes | 4,928 (100) | Web crawled |

Notably, Table 2.2 reveals a complete absence of Malaysian food datasets in the current research landscape, underscoring the significant gap that the proposed system aims to address.


Furthermore, many existing calorie estimation systems depend on international nutritional databases that do not accurately reflect the composition of local foods prepared with coconut milk, palm oil, and complex spice blends. Recent efforts to enrich MyFCD with updated sugar and macronutrient data (Norizan et al., 2025; Mognard et al., 2024) represent important steps toward addressing this limitation. However, the integration of these improved nutritional tables with culturally specific image recognition models remains underdeveloped.
 
The proposed Malaysian Hawker Food Recognition and Calorie Estimation System directly targets these gaps. By combining advanced image processing techniques for robust segmentation, a hybrid or CNN-dominant classification engine, transfer learning to handle limited local data, and direct linkage to the latest version of MyFCD, the system aims to deliver accurate food recognition and calorie estimation tailored to Malaysian dietary habits. This integrated approach aligns closely with the project’s conceptual map and is expected to provide a practical, culturally relevant tool for supporting better dietary monitoring and public health outcomes in Malaysia.
