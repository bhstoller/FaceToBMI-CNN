# Face-to-BMI Prediction System

This project aims to predict Body Mass Index (BMI) from face images. Our team implemented and compared two different approaches for BMI prediction: a VGG-Face model using TensorFlow/Keras and a ResNet50 model using PyTorch.

## Experiments

Our team conducted different experiments using complementary approaches to tackle the same problem:

|  **Approach**                          | **Framework**        | **Base Architecture** | **Key Features**                                                                 |
|----------------------------------------|----------------------|-----------------------|----------------------------------------------------------------------------------|
| Face detection + specialized model     | TensorFlow/Keras     | VGG-Face              | MTCNN face detection, command-line interface, deployment-ready structure         |
| End-to-end model                       | PyTorch              | ResNet50              | Gender feature integration, model performance analysis, comprehensive training strategy |


### VGG-Face with TensorFlow

- Model Architecture: Built on VGG-Face, a model pre-trained specifically for face recognition tasks.
- Face Detection: Implemented MTCNN to detect and extract faces from images before processing.
- Implementation: Created a modular, deployment-ready codebase with clear separation of concerns.
- User Experience: Developed both a command-line interface and a menu-driven interaction system.
- Pre-processing: Applied specialized preprocessing techniques optimized for the VGG-Face architecture.
- Deployment Focus: Structured code for easy deployment with well-defined file paths and error handling.


### ResNet50 with PyTorch

- Model Architecture: Used ResNet50, a powerful general-purpose image classification model.
- Gender Integration: Incorporated gender as an additional feature to improve prediction accuracy.
- Training Strategy: Implemented a comprehensive grid search across optimizers (Adam, SGD) and learning rates to find optimal parameters.
- Performance Analysis: Conducted in-depth analysis using multiple metrics (MSE, RMSE, MAE, MAPE, Pearson correlation).
- Data Augmentation: Applied various image transformation techniques during training to improve model generalization.
- Early Stopping: Implemented patience-based early stopping to prevent overfitting while ensuring optimal performance.

My experiments focused on model optimization and understanding the relationship between facial features, gender, and BMI through rigorous testing of different hyperparameters and model configurations.
