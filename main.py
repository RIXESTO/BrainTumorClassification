import os
import sys
import argparse
import subprocess
import tensorflow as tf

import src.config as config
from src.utils import logger, check_device
from src.dataset import get_data_loaders, compute_dataset_class_weights
from src.model import build_transfer_model
from src.train import run_two_stage_training
from src.evaluate import evaluate_model
from src.grad_cam import generate_sample_explanations

def smoke_check():
    """
    Verify all pipeline components (Data loaders, class weighting, and model graph compilation)
    without running a full training loop.
    """
    logger.info("=== 🧪 Running Pipeline Smoke Check ===")
    check_device()
    
    logger.info("1. Checking directories...")
    assert os.path.exists(config.TRAIN_DIR), f"Missing Train Dir: {config.TRAIN_DIR}"
    assert os.path.exists(config.TEST_DIR), f"Missing Test Dir: {config.TEST_DIR}"
    logger.info("✅ Dataset folders verified successfully!")

    logger.info("2. Checking Data Loaders & Class Weighting...")
    # Take a sample batch from an un-cached pipeline check to prevent dataset cache truncation warnings
    sample_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    ).take(1)
    for images, labels in sample_ds:
        logger.info(f"✅ Loaded sample batch: Images {images.shape}, Labels {labels.shape}")
        break

    train_ds, val_ds, test_ds, class_names = get_data_loaders(config)
    class_weights = compute_dataset_class_weights(config)

    logger.info("3. Checking Transfer Learning Architecture Compilation...")
    model, backbone = build_transfer_model(config)
    logger.info(f"✅ Model compiled successfully (`{model.name}`). Total parameters: {model.count_params():,}")
    
    logger.info("🎉 Smoke check passed! Everything is functional and ready for training (`python main.py --train`).")

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor MRI Classification End-to-End MLOps & XAI Pipeline")
    parser.add_argument("--check", action="store_true", help="Run a quick verification of data loaders and model compilation.")
    parser.add_argument("--train", action="store_true", help="Execute two-stage transfer learning training loop.")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model on independent test dataset.")
    parser.add_argument("--explain", action="store_true", help="Generate Grad-CAM heatmaps for sample test MRIs.")
    parser.add_argument("--all", action="store_true", help="Run Training -> Evaluation -> Grad-CAM sequentially.")
    parser.add_argument("--ui", action="store_true", help="Launch interactive Streamlit clinical diagnostic dashboard.")
    parser.add_argument("--epochs-warmup", type=int, default=None, help="Number of warmup epochs (Stage 1).")
    parser.add_argument("--epochs-finetune", type=int, default=None, help="Number of fine-tuning epochs (Stage 2).")
    
    args = parser.parse_args()

    # Print help and exit if no flags provided
    if not any([args.check, args.train, args.evaluate, args.explain, args.all, args.ui]):
        parser.print_help()
        sys.exit(0)

    if args.check:
        smoke_check()

    if args.train or args.all:
        run_two_stage_training(epochs_warmup=args.epochs_warmup, epochs_finetune=args.epochs_finetune)

    if args.evaluate or args.all:
        evaluate_model()

    if args.explain or args.all:
        generate_sample_explanations()

    if args.ui:
        logger.info("🌐 Launching Streamlit Web Application Dashboard...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
