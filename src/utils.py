import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def setup_logger(name="BrainTumorMLOps"):
    """Set up a clean logger with formatted console output."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

def check_device():
    """Check hardware acceleration available for TensorFlow."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"🚀 Hardware Acceleration Enabled: {len(gpus)} GPU(s) found -> {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Memory growth setting notice (expected on Apple Metal or pre-initialized devices): {e}")
    else:
        logger.info("ℹ️ Running on CPU (or Apple Metal Plugin if enabled natively via system TensorFlow).")

def plot_training_curves(history_dict, output_path):
    """
    Plot combined training and validation curves (Accuracy and Loss) across both Stage 1 and Stage 2.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history_dict['accuracy']) + 1)
    
    # Accuracy curve
    ax1.plot(epochs, history_dict['accuracy'], 'b-o', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, history_dict['val_accuracy'], 'r--o', label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Classification Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.set_ylim([0.4, 1.05])
    
    # Loss curve
    ax2.plot(epochs, history_dict['loss'], 'b-o', label='Training Loss', linewidth=2)
    ax2.plot(epochs, history_dict['val_loss'], 'r--o', label='Validation Loss', linewidth=2)
    ax2.set_title('Cross-Entropy Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"📊 Saved training curves to: {output_path}")

def save_metrics_summary(metrics_dict, output_path):
    """Save evaluation metrics to JSON."""
    with open(output_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logger.info(f"📁 Saved metrics summary to: {output_path}")
