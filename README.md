# Fruits & Vegetables Image-Classification (36 Classes)

A **97.5 %-accurate** transfer-learning model that distinguishes 36 kinds of fruits and vegetables using a lightweight, deployment-friendly MobileNetV2 backbone.

---

## 📸 Dataset
* **Source**  [“Fruits & Vegetables Image Recognition” – Kaggle](https://www.kaggle.com/datasets/)
* **Classes**  36 (e.g. *apple*, *banana*, *paprika*, *radish*, …)
* **Images**  ≈ 18k RGB @ 512 × 512 px
* **Splits**  70 % train | 15 % val | 15 % test (stratified)

## 🏗️ Model Architecture
| Stage       | Layer                                   | Output           | Trainable        |
|-------------|------------------------------------------|------------------|------------------|
| Backbone    | MobileNetV2 `include_top=False`          | 7 × 7 × 1280     | ❌ frozen         |
| Pooling     | GlobalAveragePooling2D                   | 1280             | ❌                |
| Regulariser | Dropout `rate=0.2`                       | —                | ❌                |
| Classifier  | Dense (36 units, softmax)                | 36               | ✔️ (46,116 params)|

*Total params: 2,396,334 (only 2 % trainable)*

## ⚙️ Training Setup
| Setting     | Value                                  |
|-------------|----------------------------------------|
| Optimiser   | Adam (lr = 3e-4, cosine decay)         |
| Loss        | Categorical cross-entropy              |
| Batch size  | 32                                     |
| Callbacks   | `EarlyStopping(patience=5)` + `ModelCheckpoint(save_best_only=True)` |
| Hardware    | NVIDIA RTX 3060 (12 GB)                |

## 📈 Results
| Metric        | Value     |
|---------------|-----------|
| Test accuracy | **97.49%**|
| Test loss     | 0.1492    |

![Learning Curves](assets/fig_curves.png)
*Training & validation accuracy/loss (18 epochs, early-stop at epoch 18)*

![Confusion Matrix](assets/fig_confusion_matrix.png)
*Normalised confusion matrix (36 classes)*

### Error Analysis

| True → Predicted        | Cause                                |
|-------------------------|---------------------------------------|
| Banana → Ginger/Chilli  | Elongated cartoon silhouette          |
| Apple (logo) → Paprika  | Stylised rainbow colour palette       |
| Corn ↔ Sweetcorn        | Close-up kernels vs whole cob         |
| Carrot (clip-art) → Radish | Simplified tapered shape          |

See: `assets/fig_misclassified.png` for examples.

---

## 🚀 Quick Start
```bash
# 1) Clone
git clone https://github.com/<your-username>/fruits-veggies-classifier.git
cd fruits-veggies-classifier

# 2) Setup environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) Download dataset (≈2 GB) into /data

# 4) Train model
python train.py

# 5) Evaluate
python evaluate.py --weights runs/best_model.h5
