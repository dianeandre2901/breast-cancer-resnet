
# 🧠 Breast Cancer Image Classification with ResNet50

This project fine-tunes a pretrained ResNet50 model to classify histopathology images of breast cancer into benign or malignant types.  
Built using PyTorch and trained on the BreakHis dataset (converted to `.npy` format).

- 📦 Framework: PyTorch
- 📈 Techniques: Transfer learning, data augmentation, early stopping, ROC analysis
- 🧪 Metrics: Accuracy, F1-score, ROC-AUC
  
## 🔬 Method
1. Preprocessed `.npy` histopathology images
2. Fine-tuned `ResNet50` with new classification head
3. Used learning rate scheduler, early stopping, data augmentation and class weights
4. Evaluated with F1-score, confusion matrix, and ROC-AUC

## 📥 Download Pretrained Model

You can download the fine-tuned ResNet50 model (`best_model.pt`) from the link below:

🔗 https://drive.google.com/file/d/1qgGBZacJ3TrvERhNy2B0x5Upk__DWJRU/view?usp=drive_link

After downloading, place the file in the `models/` folder:


## 🧪 Final Test Set Performance

This model was evaluated on a held-out test set after hyperparameter tuning and early stopping.  
Performance metrics were computed from the final confusion matrix and ROC-AUC curve.

| Metric     | Value     |
|------------|-----------|
| Accuracy   | **95.18%** |
| Precision  | **98.14%** |
| Recall     | **95.26%** |
| F1 Score   | **96.68%** |
| AUC        | **95.11%** |


📊 Confusion Matrix:

<img src="results/confusion_matrix.png" width="400">  

📈 Training and Validation curve:

<img src="results/training_val_curve.png" width="400">

> 🎯 This model achieved high precision and recall, indicating strong ability to distinguish benign from malignant samples — a critical need in medical diagnosis.

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
python src/utils.py --model models/best_model.pt
