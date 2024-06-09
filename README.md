# Object-detection

## Project Overview

This project aims to collect and annotate images containing Spiderman and other individuals. The annotated images are then used to fine-tune a pre-trained YOLOv8 model for object detection. The trained model is capable of accurately identifying and localizing Spiderman and other people in both images and videos.

### Introduction

This project is designed to detect Spiderman and other people in images and videos using a fine-tuned YOLOv8 model. The process involves collecting relevant images, annotating them, training the model, and visualizing the results.

### Data Collection

The initial step involves gathering images that include Spiderman and other individuals. These images serve as the independent variables for the model training process. 

### Annotation

The collected images are annotated using the CVAT (Computer Vision Annotation Tool). Annotations include bounding boxes around Spiderman and other individuals. These annotated images form the target data for model training.

### Model Training

After annotation, the images are formatted to be compatible with the YOLOv8 model. The pre-trained YOLOv8 model is then fine-tuned using the annotated dataset. This process involves:

Preparing the annotated images for training.
Feeding the prepared images into the YOLOv8 model.
Fine-tuning the model to improve detection accuracy.
Obtaining the best.pt model which contains the optimized weights.

### Visualization

Using the trained best.pt model, we utilize OpenCV (cv2) to visualize the predictions. The model outputs images with bounding boxes around detected objects (Spiderman and other people). While this project primarily focuses on images, the model is also effective for video analysis.

### Conclusion

This project demonstrates an end-to-end pipeline for collecting, annotating, and training a YOLOv8 model for detecting Spiderman and other individuals. The fine-tuned model performs well on both images and videos, providing accurate and reliable object detection.
