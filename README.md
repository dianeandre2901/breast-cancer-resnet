# 🧠 Breast Cancer Image Classifier with ResNet50

This project uses transfer learning with ResNet50 to classify breast cancer subtypes from histopathology images.

## 💡 Overview
- **Goal**: Multi-class classification of breast cancer subtypes
- **Data**: BreakHis dataset (benign vs malignant + subtypes)
- **Model**: ResNet50 pretrained on ImageNet
- **Tools**: PyTorch, torchvision, matplotlib, numpy

## 🔬 Method
1. Preprocessed `.npy` histopathology images
2. Fine-tuned `ResNet50` with new classification head
3. Used learning rate scheduler, early stopping, data augmentation and class weights
4. Evaluated with F1-score, confusion matrix, and ROC-AUC

## 🧪 Results

<img src="results/cm_resnet.png" width="400">
<img src="results/roc_auc.png" width="400">

- 📈 Accuracy: 87%
- 🔁 F1-Score (macro): 0.85
- 🩺 Strong recall on malignant subtype classes

## 🧠 Key Learnings
- Transfer learning is highly effective on medical images
- Balanced validation and early stopping helped generalize better
- Visualizing confusion matrix revealed subtype misclassification patterns

## 🚀 How to Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training
python src/train.py

# 3. Evaluate or load model
python src/evaluate.py --model models/best_model.pt
