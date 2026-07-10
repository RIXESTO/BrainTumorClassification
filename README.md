# 🧠 NeuroScan AI — Brain Tumor MRI Classification & Explainable AI (XAI) Suite

An end-to-end, medical-grade Deep Learning and MLOps pipeline for **multi-class Brain Tumor MRI Classification** (`Glioma`, `Meningioma`, `Pituitary`, and `No Tumor`). Incorporates domain-safe data augmentations, two-stage transfer learning (EfficientNetB0 / ResNet50V2), real-time Grad-CAM spatial activation mapping, quantitative clinical evaluation reporting, and an interactive **Streamlit Clinical Diagnostic Dashboard**.

---

## 🌟 Key Features

1. **🏥 Medical-Grade Transfer Learning Architecture (`src/model.py`)**
   - **Backbones:** Built on `EfficientNetB0` (default) and `ResNet50V2` initialized with ImageNet weights.
   - **Two-Stage Fine-Tuning Strategy (`src/train.py`):**
     - *Stage 1 (Warmup):* Freezes the backbone and trains a custom, regularized multi-class classification head (`GlobalAveragePooling2D` -> `BatchNormalization` -> `Dropout(0.5)` -> `Dense(256)` -> `Dropout(0.3)` -> Softmax).
     - *Stage 2 (End-to-End Fine-Tuning):* Unfreezes top layers while **strictly maintaining `BatchNormalization` layers frozen** (`layer.trainable = False`) to protect learned batch statistics and prevent catastrophic forgetting.
   - **Imbalance Handling:** Automatically computes exact inverse class weights (`compute_class_weight='balanced'`) across `TRAIN_DIR` to eliminate dataset bias toward the `no_tumor` class.

2. **🔍 Explainable AI (XAI) Engine (`src/grad_cam.py`)**
   - **Gradient-weighted Class Activation Mapping (Grad-CAM):** Extracts spatial feature activations from the top convolutional layers (`top_conv`).
   - **Exact Localization:** Computes exact channel gradients via `tf.GradientTape()` and applies ReLU activation (`tf.maximum(heatmap, 0)`) to highlight only positive spatial contributions toward the predicted tumor pathology.
   - **Numerical Safeguards:** Features `axis=-1` tensor squeezing (`tf.squeeze(heatmap, axis=-1)`) to guarantee stable 2D spatial dimensions even on 1x1 feature maps, robust `grads is None` defensive checks, and exact `np.clip` colormap superimposition (`cv2.COLORMAP_JET`).

3. **📊 Exhaustive Clinical Evaluation Suite (`src/evaluate.py`)**
   - **One-vs-Rest ROC & AUC Curves:** Computes independent Receiver Operating Characteristic curves and exact Area Under the Curve (`AUC`) across all 4 clinical categories.
   - **High-Resolution Confusion Matrix:** Exports annotated seaborn heatmaps (`confusion_matrix.png`).
   - **Full Diagnostic Metrics Export:** Exports complete precision, recall (sensitivity), F1-scores, and support breakdown to JSON (`test_evaluation_report.json`).

4. **🌐 Interactive Streamlit Diagnostic Dashboard (`app.py`)**
   - **Clinical UI:** A responsive, modern dashboard featuring model caching (`@st.cache_resource`), real-time Grad-CAM opacity adjustments (`alpha_val`), sample test dataset pickers, custom `.png/.jpg/.jpeg` uploaders, and pathology breakdowns.

---

## 📁 Repository Structure

```text
├── BrainTumorMRI/               # MRI Image Dataset (Training & Testing splits)
│   ├── Training/                # 2,870 MRI Scans (glioma, meningioma, no_tumor, pituitary)
│   └── Testing/                 # 394 Independent Test Scans
├── models/                      # Saved Model Checkpoints
│   └── best_brain_tumor_model.keras
├── outputs/                     # Generated Diagnostics & Visualizations
│   ├── confusion_matrix.png
│   ├── grad_cam_samples.png
│   ├── roc_curves.png
│   └── test_evaluation_report.json
├── src/                         # Core Source Code Package
│   ├── __init__.py
│   ├── config.py                # Hyperparameters, paths & class definitions
│   ├── dataset.py               # tf.data.Dataset pipelines, augmentations & class weights
│   ├── model.py                 # Transfer learning architecture & fine-tuning logic
│   ├── train.py                 # Two-stage training loop & callback management
│   ├── evaluate.py              # Quantitative metrics & ROC-AUC generation
│   ├── grad_cam.py              # Grad-CAM XAI engine & diagnostic grid generator
│   └── utils.py                 # Hardware acceleration check, logging & plotting helpers
├── app.py                       # Streamlit Web Diagnostic Application
├── main.py                      # CLI Entrypoint & Pipeline Orchestrator
├── requirements.txt             # Project Dependencies
└── README.md                    # Project Documentation
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
Make sure you have Python 3.10+ installed. Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity (Smoke Check)
Run a fast, un-cached verification of dataset directories, class weights, and model compilation without triggering training:
```bash
python main.py --check
```

### 3. Execute Two-Stage Transfer Learning Loop
Train the classification model with automated checkpoints (`ModelCheckpoint`), early stopping (`EarlyStopping`), and learning rate scheduling (`ReduceLROnPlateau`):
```bash
python main.py --train
```

### 4. Run Independent Test Evaluation
Generate multi-class classification reports, annotated confusion matrices, and One-vs-Rest ROC-AUC curves:
```bash
python main.py --evaluate
```

### 5. Generate Multi-Class XAI Heatmaps
Create visual Grad-CAM overlays across all tumor categories and export high-resolution diagnostic grids (`outputs/grad_cam_samples.png`):
```bash
python main.py --explain
```

### 6. Launch Streamlit Diagnostic Web Application
Start the interactive clinical diagnostic dashboard:
```bash
python main.py --ui
# Or directly via streamlit:
streamlit run app.py
```

---

## 🔬 Dataset & Clinical Classes

The dataset consists of axial, coronal, and sagittal T1-weighted contrast-enhanced magnetic resonance imaging (MRI) scans classified into four distinct diagnostic targets:
1. `glioma_tumor`: Primary tumors originating in glial brain/spinal tissue.
2. `meningioma_tumor`: Typically benign slow-growing tumors originating from the meningeal membranes.
3. `pituitary_tumor`: Adenomas developing in the pituitary gland at the cranial base.
4. `no_tumor`: Healthy brain MRI scans exhibiting no pathological mass or architectural distortion.

---

## ⚖️ License
This project is for educational, research, and technical demonstration purposes. Medical AI outputs generated by this suite should not be used as sole determinants for clinical diagnosis without certified radiological review.
