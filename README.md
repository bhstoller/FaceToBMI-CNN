# Face-to-BMI Prediction System

This project aims to predict Body Mass Index (BMI) from face images. Our team implemented and compared two different approaches for BMI prediction: a VGG-Face model using TensorFlow/Keras and a ResNet50 model using PyTorch.

## Experiments

Our team conducted different experiments using complementary approaches to tackle the same problem:

|  **Approach**                          | **Framework**        | **Base Architecture** | **Key Features**                                                                 | **Pearson r Correlation** |
|----------------------------------------|----------------------|-----------------------|----------------------------------------------------------------------------------|---------------|
| Face detection + specialized model     | TensorFlow/Keras     | VGG-Face              | MTCNN face detection, command-line interface, deployment-ready structure         | |
| End-to-end model                       | PyTorch              | ResNet50              | Gender feature integration, model performance analysis, comprehensive training strategy | 0.661 |
| ViT-based regression model             | PyTorch + HuggingFace| ViT-B/16-IN21k        | Fine-tuned on facial images, model performance, intense training | 0.685|
| Landmark + CNN ensemble                | PyTorch              | MLP + MobileNetV3     | Combines landmark MLP and CNN predictions, moderate gains over individual models  | 0.584 |



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

### Vision Transformer (ViT) with PyTorch + Hugging Face
- **Model Architecture**: Fine-tuned `vit-base-patch16-224-in21k`, a pretrained Vision Transformer trained on ImageNet-21k.
- **Training Strategy**: Used custom regression head with ReLU and dropout, optimized with AdamW and `ReduceLROnPlateau` scheduler.
- **Preprocessing**: Applied Hugging Face `ViTFeatureExtractor`-based transform for resizing, normalization, and tensor conversion.
- **Performance**: Achieved Pearson r = **0.688**, outperforming both baseline and paper-reported results.
- **Evaluation Metrics**: Tracked MAE, R², and Pearson correlation across epochs to assess model learning and stability.

### Ensemble: Landmark MLP + CNN
- **Architecture**: Combined predictions from a facial-landmark-based MLP and a MobileNetV3 CNN trained on facial images.
- **Ensembling Method**: Blended model outputs using weighted averaging to test different contribution levels.
- **Performance Outcome**: Did not outperform the ViT model, but demonstrated moderate standalone performance for the MLP.
- **Purpose**: Explored complementary strength of image-based and structured (landmark) inputs for BMI regression.


