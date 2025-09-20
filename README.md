# Rice Disease Detection Using ResNet-18

A deep learning model for detecting rice plant diseases using computer vision. This project implements a ResNet-18 based classifier to identify three common rice diseases: Brown Spot, Bacterial Leaf Blight, and Leaf Scald.

## 🌾 Overview

Rice is one of the world's most important food crops, and early disease detection is crucial for maintaining crop health and yield. This project uses transfer learning with a pre-trained ResNet-18 model to automatically classify rice leaf diseases from images.

## 🎯 Features

- **Multi-class Classification**: Detects 3 types of rice diseases
- **Transfer Learning**: Uses pre-trained ResNet-18 for improved performance
- **Data Augmentation**: Includes rotation, flipping, and color jittering
- **Comprehensive Evaluation**: Provides detailed metrics and visualizations
- **Single Image Prediction**: Test model on individual images
- **Model Optimization**: Includes learning rate scheduling and early stopping

## 📊 Supported Diseases

1. **Brown Spot** - Fungal disease causing brown lesions on leaves
2. **Bacterial Leaf Blight** - Bacterial infection affecting leaf edges
3. **Leaf Scald** - Disease causing scalding appearance on leaves

## 🛠️ Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
Pillow>=8.3.0
tqdm>=4.62.0
torchsummary>=1.5.1
kaggle>=1.5.12
```

## 📁 Project Structure

```
rice-disease-detection/
├── README.md
├── requirements.txt
├── rice_disease_detection.ipynb
├── best_rice_disease_model.pth
└── data/
    └── Rice_Leaf_AUG/
        ├── Brown Spot/
        ├── Bacterial Leaf Blight/
        └── Leaf scald/
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install required packages
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn Pillow tqdm torchsummary kaggle
```

### 2. Dataset Preparation

The model uses the Rice Disease Dataset from Kaggle. The code automatically downloads and sets up the dataset:

```python
# The code handles dataset download automatically
!kaggle datasets download -d anshulm257/rice-disease-dataset
```

**Note**: You'll need to upload your `kaggle.json` API credentials file when prompted.

### 3. Training the Model

Run the provided Jupyter notebook cells in sequence:

1. **Setup and Install** - Install required packages
2. **Data Loading** - Download and prepare the dataset
3. **Data Preprocessing** - Apply transformations and augmentation
4. **Model Definition** - Create ResNet-18 based classifier
5. **Training** - Train the model with validation
6. **Evaluation** - Test model performance and generate reports

### 4. Using Pre-trained Model

```python
# Load the trained model
model = RiceDiseaseResNet18(num_classes=3, pretrained=True)
checkpoint = torch.load('best_rice_disease_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Make prediction on new image
predicted_class, confidence = predict_single_image(
    model, 'path/to/image.jpg', test_transform, class_names, device
)
```

## 🧠 Model Architecture

- **Base Model**: ResNet-18 (pre-trained on ImageNet)
- **Input Size**: 224×224×3 RGB images
- **Output**: 3 classes (disease types)
- **Custom Classifier**: 
  - Dropout (0.3)
  - Linear layer (512 → 256)
  - ReLU activation
  - Dropout (0.2)
  - Output layer (256 → 3)

## 📈 Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross-Entropy Loss
- **Batch Size**: 32
- **Epochs**: 25
- **Data Split**: 70% train, 15% validation, 15% test
- **Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Based on validation accuracy

## 🎨 Data Augmentation

Training images undergo the following augmentations:
- Random horizontal flip (50%)
- Random vertical flip (30%)
- Random rotation (±20°)
- Color jittering (brightness, contrast, saturation, hue)
- Random crop (224×224 from 256×256)

## 📊 Model Performance

The model provides:
- **Training/Validation Loss and Accuracy Curves**
- **Confusion Matrix** with per-class accuracy
- **Classification Report** with precision, recall, F1-score
- **Inference Time Analysis**
- **Model Size and Parameter Count**

## 🔍 Inference

### Single Image Prediction

```python
# Predict disease from image
predicted_class, confidence = predict_single_image(
    model=model,
    image_path="path/to/rice_leaf.jpg",
    transform=test_transform,
    class_names=target_diseases,
    device=device
)

print(f"Predicted Disease: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")
```

### Batch Prediction

```python
# Process multiple images
for image_path in image_list:
    prediction = model_predict(image_path)
    print(f"{image_path}: {prediction}")
```

## 🏗️ Model Statistics

- **Total Parameters**: ~11M parameters
- **Model Size**: ~44 MB
- **Average Inference Time**: ~10-50ms per image (GPU)
- **Input Resolution**: 224×224 pixels

## 📝 Usage Examples

### Training from Scratch

```python
# Initialize model
model = RiceDiseaseResNet18(num_classes=3, pretrained=True)

# Train the model
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
```

### Evaluation

```python
# Generate comprehensive evaluation
test_loss, test_acc, predictions, labels = validate_epoch(model, test_loader, criterion, device)
print(classification_report(labels, predictions, target_names=class_names))
```

## 🔧 Customization

### Adding New Disease Classes

1. Update the dataset folder structure
2. Modify `num_classes` parameter
3. Update `target_diseases` list
4. Retrain the model

### Adjusting Model Architecture

```python
# Modify the classifier head
self.resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)
```

## ⚠️ Important Notes

- **GPU Recommended**: Training is significantly faster with CUDA-enabled GPU
- **Data Quality**: Ensure high-quality, well-labeled training images
- **Class Balance**: The model performs best with balanced class distributions
- **Image Format**: Supports PNG, JPG, JPEG, BMP formats
- **Memory Requirements**: ~2-4GB GPU memory for training

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Rice Disease Dataset from Kaggle
- PyTorch team for the framework
- ResNet paper authors for the architecture
- ImageNet for pre-trained weights

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Happy farming! 🌾🤖**
