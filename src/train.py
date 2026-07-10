import os
import tensorflow as tf
import src.config as config
from src.dataset import get_data_loaders, compute_dataset_class_weights
from src.model import build_transfer_model, unfreeze_backbone_for_finetuning
from src.utils import logger, check_device, plot_training_curves, save_metrics_summary

def run_two_stage_training(epochs_warmup=None, epochs_finetune=None):
    """
    Execute the full two-stage training loop:
      - Stage 1: Train custom classification head with frozen EfficientNet/ResNet backbone.
      - Stage 2: Unfreeze top layers and fine-tune with low learning rate.
    Uses class weights to eliminate dataset bias on 'no_tumor' class.
    """
    check_device()
    
    if epochs_warmup is None:
        epochs_warmup = config.WARMUP_EPOCHS
    if epochs_finetune is None:
        epochs_finetune = config.FINETUNE_EPOCHS

    logger.info("=== STEP 1: Loading Data Pipelines & Computing Class Weights ===")
    train_ds, val_ds, test_ds, class_names = get_data_loaders(config, val_split=0.2)
    class_weights = compute_dataset_class_weights(config)

    logger.info("=== STEP 2: Initializing Model & Callbacks ===")
    model, backbone = build_transfer_model(config)

    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    logger.info(f"=== STEP 3: Stage 1 Training (Warmup Head - {epochs_warmup} Epochs) ===")
    history_s1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_warmup,
        class_weight=class_weights,
        callbacks=callbacks
    )

    logger.info(f"=== STEP 4: Stage 2 Fine-Tuning ({epochs_finetune} Epochs) ===")
    model = unfreeze_backbone_for_finetuning(
        model, 
        backbone, 
        num_layers_to_unfreeze=30, 
        learning_rate=config.LEARNING_RATE_STAGE2
    )

    actual_epochs_s1 = len(history_s1.history['accuracy'])
    history_s2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=actual_epochs_s1 + epochs_finetune,
        initial_epoch=actual_epochs_s1,
        class_weight=class_weights,
        callbacks=callbacks
    )

    logger.info("=== STEP 5: Aggregating History & Generating Diagnostics Curves ===")
    combined_history = {
        key: history_s1.history.get(key, []) + history_s2.history.get(key, [])
        for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']
        if key in history_s1.history and key in history_s2.history
    }

    curves_path = os.path.join(config.OUTPUTS_DIR, "training_curves.png")
    plot_training_curves(combined_history, curves_path)

    summary_path = os.path.join(config.OUTPUTS_DIR, "training_metrics.json")
    save_metrics_summary({
        'best_val_accuracy': max(combined_history['val_accuracy']),
        'best_val_loss': min(combined_history['val_loss']),
        'total_epochs_trained': len(combined_history['accuracy'])
    }, summary_path)

    logger.info(f"🎉 Training Pipeline Complete! Best model saved at: {config.MODEL_SAVE_PATH}")
    return model, combined_history

if __name__ == '__main__':
    run_two_stage_training()
