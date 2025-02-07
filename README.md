# Modified-CGrabcut-Coffee-Leaf-Disease

## Introduction
This project focuses on modifying the C-Grabcut algorithm to improve the segmentation and classification of coffee leaf diseases in images with complex backgrounds. The primary goal is to develop a robust method that can accurately identify and classify various diseases affecting coffee leaves, even when the background of the image is cluttered or complex.

## Objectives
- Enhance the C-Grabcut Algorithm: Modify the existing C-Grabcut algorithm to improve its performance in segmenting coffee leaf images.
- Disease Classification: Implement Deep Learning techniques to classify different types of diseases affecting coffee leaves.
- Complex Background Handling: Ensure the algorithm performs well even when the background of the images is complex and cluttered.
- Accuracy and Efficiency: Achieve high accuracy and efficiency in both segmentation and classification tasks.

## Methodology
1. Data Collection: Gather a dataset of coffee leaf images from public dataset, including healthy leaves and leaves affected by various diseases (data source: https://universe.roboflow.com/tugas-akhir-70fw5/deteksi-penyakit-daun-kopi-robusta).
2. Preprocessing: Apply preprocessing techniques to enhance the quality of the images and prepare them for segmentation.
3. Algorithm Modification: Modify the C-Grabcut algorithm to improve its performance in segmenting coffee leaf regions from complex backgrounds.
4. Feature Extraction: Extract relevant features from the segmented leaf regions to aid in the classification of diseases.
5. Classification: Use deep learning models (Transfer Learning MobileNet-V2) to classify the diseases.
6. Evaluation: Evaluate the performance of the modified algorithm and classification models using metrics such as accuracy, precision, recall, and F1-score.

## Results
The modified C-Grabcut algorithm demonstrated significant improvements in segmenting coffee leaf images with complex backgrounds. The classification models achieved high accuracy in identifying and classifying various coffee leaf diseases.

## Dataset link
Pre-processed dataset can be download here : https://app.roboflow.com/tesis-ihiwj/coffee-leaves-disease/1

## References
- Anonim, Deteksi Penyakit Daun Kopi Robusta Dataset. Roboflow, 2023.
- K. Kusrini et al., “Data augmentation for automated pest classification in Mango farms,” Comput Electron Agric, vol. 179, Dec. 2020, doi: 10.1016/j.compag.2020.105842.
- S. Lian, L. Guan, J. Pei, G. Zeng, and M. Li, “Identification of apple leaf diseases using C-Grabcut algorithm and improved transfer learning base on low shot learning,” Multimed Tools Appl, vol. 83, no. 9, pp. 27411–27433, Mar. 2024, doi: 10.1007/s11042-023-16602-4.
