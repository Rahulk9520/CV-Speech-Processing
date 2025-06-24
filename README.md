# Computer Vision and Speech Processing

**Advanced Computer Vision and Speech Processing Techniques using Python, OpenCV, scikit-learn, and Librosa**  
*By Rahul Kumar*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Task List (CV1–CV13)](#task-list-cv1cv13)
- [Directory Structure](#directory-structure)
- [Sample Code Snippets](#sample-code-snippets)
- [References](#references)
- [Contact](#contact)

---

## Project Overview

This repository contains a series of advanced tasks and mini-projects in computer vision and speech processing, including image preprocessing, feature extraction, object and face detection, image recognition, and speech-based classification. The work demonstrates practical applications of deep learning and classical machine learning using real-world datasets and Python-based toolkits.

---

## Motivation

Computer vision and speech processing are at the heart of modern AI applications, enabling machines to interpret visual and auditory information. This project aims to provide hands-on experience with foundational and state-of-the-art algorithms for image and speech analysis, recognition, and classification.

---

## Task List (CV1–CV13)

### CV1: Colour Conversion and Geometric Transformations
- Convert images between colour spaces (BGR, HSV, grayscale).
- Perform resizing, translation, rotation, and affine transformations.

### CV2: Image Filtering and Edge Detection
- Apply averaging, Gaussian, median, and bilateral filters.
- Detect edges using Sobel, Canny, and custom convolution kernels.
- Use morphological operations (opening, closing, dilation, erosion).

### CV3: Image Histograms and Similarity Measures
- Compute and plot histograms for RGB and grayscale images.
- Perform histogram equalization for contrast enhancement.
- Compare images using similarity metrics (Chi-Square, KL-Divergence).

### CV4: Connected Component Labelling and Morphological Operations
- Identify and label connected components in binary images.
- Apply morphological operations for noise removal and object separation.

### CV5: Feature Extraction and Matching (SIFT, Harris, Hausdorff)
- Detect keypoints using SIFT and Harris corner detectors.
- Extract descriptors and match features between images.
- Use Hausdorff distance for shape comparison.

### CV6: Affine Invariance and Keypoint Matching
- Test affine invariance of feature detectors.
- Match keypoints across rotated, scaled, or transformed images.

### CV7: Face Detection and Parameter Tuning
- Implement face detection using Haar cascades.
- Tune parameters for robust detection in various conditions.

### CV8: Pedestrian Detection and Non-Maximum Suppression
- Detect pedestrians in images using HOG + SVM.
- Apply non-maximum suppression to refine bounding boxes.

### CV9: Real-time Object Detection with Webcam
- Integrate webcam feed for real-time face or object detection.
- Draw bounding boxes and annotate live video streams.

### CV10: Image Recognition using Bag of Words (BoW)
- Build a visual dictionary using SIFT + KMeans clustering.
- Classify images (e.g., food categories) using BoW and classifiers (k-NN, SVM, AdaBoost).
- Evaluate with confusion matrices and accuracy metrics.

### CV11: Speech Feature Extraction (MFCC) and Speaker Recognition
- Extract MFCC features from audio signals using Librosa.
- Train Gaussian Mixture Models (GMMs) for speaker identification.
- Evaluate with classification reports and confusion matrices.

### CV12: Speech Emotion Recognition
- Extract MFCC features from emotion-labeled audio.
- Train SVM and AdaBoost classifiers for emotion recognition.
- Evaluate with accuracy, precision, recall, and F1-score.

### CV13: Project Integration and Final Evaluation
- Integrate computer vision and speech processing pipelines.
- Summarize results, compare models, and discuss practical applications.
- Prepare final presentations and documentation.

---

## Directory Structure

```
Computer-Vision-and-Speech-Processing/
│
├── data/                                      # Datasets and resources (images, audio)
├── notebooks/                                 # Jupyter notebooks for each CV task (CV1–CV13)
│ ├── CV1_colour_conversion.ipynb
│ ├── CV2_filtering_edge_detection.ipynb
│ ├── CV3_histograms_similarity.ipynb
│ ├── CV4_connected_components.ipynb
│ ├── CV5_feature_extraction.ipynb
│ ├── CV6_affine_invariance.ipynb
│ ├── CV7_face_detection.ipynb
│ ├── CV8_pedestrian_detection.ipynb
│ ├── CV9_realtime_detection.ipynb
│ ├── CV10_bow_image_recognition.ipynb
│ ├── CV11_speech_mfcc_gmm.ipynb
│ ├── CV12_speech_emotion_recognition.ipynb
│ └── CV13_integration_evaluation.ipynb
├── src/                                       # Source code scripts (preprocessing, feature extraction, modeling)
├── results/                                   # Output images, plots, and model files
├── requirements.txt                           # Python dependencies
├── README.md                                  # Project documentation
└── LICENSE
```

---

## Sample Code Snippets

### Colour Conversion & Geometric Transformations (CV1)

```python
import cv2 as cv
img = cv.imread('img1.jpg')
img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_resize = cv.resize(img, (100, 100))
```

### Feature Extraction & Matching (SIFT, CV5)

```python
sift = cv.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img_gray, None)
bf = cv.BFMatcher()
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
```

### MFCC Extraction for Speech (CV11, CV12)

```python
import librosa
y, sr = librosa.load('audio.wav')
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
```

---

## References

1. [OpenCV Documentation](https://docs.opencv.org/)
2. [scikit-learn Documentation](https://scikit-learn.org/)
3. [Librosa Documentation](https://librosa.org/doc/latest/index.html)
4. [Matplotlib Documentation](https://matplotlib.org/)
5. [Python Official Documentation](https://docs.python.org/3/)

---

## Contact

**Author:** Rahul Kumar  
**Email:** kumar.rahul226@gmail.com  
**LinkedIn:** [rk95-dataquasar](https://www.linkedin.com/in/rk95-dataquasar/)

