import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import src.config as config
from src.dataset import get_data_loaders
from src.utils import logger, check_device

def evaluate_model(model_path=None):
    """
    Perform medical-grade quantitative evaluation on the independent test dataset:
      - Overall Test Accuracy & Loss
      - Macro/Micro Precision, Recall (Sensitivity), and F1-Score per class
      - High-resolution annotated Confusion Matrix
      - Multi-class One-vs-Rest ROC Curves and AUC metrics
    """
    check_device()
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        logger.error(f"❌ Model checkpoint not found at: {model_path}. Please run training first via `python main.py --train`.")
        return None

    logger.info(f"📂 Loading trained checkpoint from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    _, _, test_ds, class_names = get_data_loaders(config)
    
    logger.info("🧪 Evaluating model with Test-Time Augmentation (TTA) across the independent Test Dataset (394 images)...")
    y_true = []
    y_pred_probs = []

    # TTA augmentations: original + flipped + brightness variants
    def tta_predict(model, images, n_augments=5):
        """Average predictions over original + augmented variants for robust inference."""
        all_preds = []
        # 1. Original
        all_preds.append(model.predict(images, verbose=0))
        # 2. Horizontal flip
        all_preds.append(model.predict(tf.image.flip_left_right(images), verbose=0))
        # 3. Slight brightness increase (simulates brighter test images)
        bright_up = tf.clip_by_value(images * 1.15, 0, 255)
        all_preds.append(model.predict(bright_up, verbose=0))
        # 4. Slight brightness decrease
        bright_down = tf.clip_by_value(images * 0.85, 0, 255)
        all_preds.append(model.predict(bright_down, verbose=0))
        # 5. Slight contrast adjustment
        mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
        contrast = tf.clip_by_value((images - mean) * 1.1 + mean, 0, 255)
        all_preds.append(model.predict(contrast, verbose=0))
        return np.mean(all_preds, axis=0)

    for images, labels in test_ds:
        preds = tta_predict(model, images)
        y_true.extend(labels.numpy())
        y_pred_probs.extend(preds)

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    logger.info(f"🎯 Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    # 1. Classification Report
    report_dict = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)
    report_str = classification_report(y_true, y_pred_classes, target_names=class_names, digits=4)
    print("\n" + "="*60)
    print("📋 CLINICAL CLASSIFICATION REPORT (Independent Test Set)")
    print("="*60)
    print(report_str)
    print("="*60 + "\n")

    # 2. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar_kws={'label': 'Number of MRI Scans'},
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title("Confusion Matrix — Brain Tumor Classification", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Predicted Tumor Class", fontsize=13, labelpad=10)
    plt.ylabel("Ground Truth Tumor Class", fontsize=13, labelpad=10)
    plt.xticks(rotation=25, ha="right", fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(config.OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logger.info(f"📊 Saved Confusion Matrix to: {cm_path}")

    # 3. Multi-Class ROC Curves (One-vs-Rest)
    y_true_bin = label_binarize(y_true, classes=range(config.NUM_CLASSES))
    plt.figure(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    roc_auc_dict = {}

    for i in range(config.NUM_CLASSES):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            roc_auc_dict[class_names[i]] = float(roc_auc)
            plt.plot(
                fpr, 
                tpr, 
                color=colors[i % len(colors)], 
                lw=2.5,
                label=f"{class_names[i]} (AUC = {roc_auc:.4f})"
            )
        else:
            logger.warning(f"⚠️ Class '{class_names[i]}' has only one class label present in ground truth. Skipping ROC curve.")
            roc_auc_dict[class_names[i]] = 0.0

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.5)")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=13)
    plt.title("Multi-Class ROC Curves — One-vs-Rest Diagnostic Evaluation", fontsize=15, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(config.OUTPUTS_DIR, "roc_curves.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    logger.info(f"📈 Saved ROC-AUC Curves to: {roc_path}")

    # Save full evaluation summary to JSON
    eval_summary = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "roc_auc_per_class": roc_auc_dict,
        "classification_report": report_dict
    }
    report_json_path = os.path.join(config.OUTPUTS_DIR, "test_evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(eval_summary, f, indent=4)
    logger.info(f"📁 Saved comprehensive evaluation JSON to: {report_json_path}")

    return eval_summary

if __name__ == "__main__":
    evaluate_model()
