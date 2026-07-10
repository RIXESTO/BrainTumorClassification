import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from src.utils import logger

def get_data_loaders(config, val_split=0.2):
    """
    Load brain tumor MRI images from disk into optimized tf.data.Dataset pipelines
    with automatic 80/20 train/validation split and a separate test dataset.
    """
    if not os.path.exists(config.TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {config.TRAIN_DIR}. Please verify dataset path.")
    if not os.path.exists(config.TEST_DIR):
        raise FileNotFoundError(f"Testing directory not found: {config.TEST_DIR}. Please verify dataset path.")

    logger.info(f"📁 Loading Training dataset from: {config.TRAIN_DIR}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=val_split,
        subset="training",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    )

    logger.info(f"📁 Loading Validation dataset from: {config.TRAIN_DIR}")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        validation_split=val_split,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int"
    )

    logger.info(f"📁 Loading Testing dataset from: {config.TEST_DIR}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        seed=config.SEED,
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        label_mode="int",
        shuffle=False
    )

    # Verify class names match our expectation
    class_names = train_ds.class_names
    logger.info(f"🏷️ Detected Classes ({len(class_names)}): {class_names}")

    # Optimize datasets for fast execution (caching & asynchronous prefetching)
    # Note: Applying .shuffle() AFTER .cache() on train_ds ensures that on every training epoch,
    # the cached mini-batches are presented in a freshly randomized order.
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(buffer_size=1000, seed=config.SEED).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

def get_augmentation_layer():
    """
    Build domain-safe data augmentation layers specifically for Brain MRI scans.
    Prevents overfitting on small medical datasets without distorting anatomy.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),    # ~15 degree subtle tilt
        tf.keras.layers.RandomZoom(0.1),        # Slight scale variation
        tf.keras.layers.RandomContrast(0.1)     # Slight lighting variation across MRI machines
    ], name="mri_data_augmentation")
    return data_augmentation

def compute_dataset_class_weights(config):
    """
    Compute inverse class weights to address dataset imbalance.
    Specifically balances 'no_tumor' (395 samples) with tumor classes (~825 samples).
    """
    class_counts = []
    class_indices = []
    
    for idx, cls_name in enumerate(config.CLASS_NAMES):
        cls_folder = os.path.join(config.TRAIN_DIR, cls_name)
        if os.path.exists(cls_folder):
            count = len([f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts.append(count)
            class_indices.append(idx)
        else:
            logger.warning(f"Class folder not found: {cls_folder}")
            class_counts.append(1)
            class_indices.append(idx)

    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    
    # Standard formula: n_samples / (n_classes * n_samples_j)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.array(class_indices),
        y=np.repeat(class_indices, class_counts)
    )
    
    class_weight_dict = {i: float(weight) for i, weight in enumerate(class_weights_array)}
    logger.info(f"⚖️ Computed Class Weights for imbalanced dataset:")
    for idx, weight in class_weight_dict.items():
        logger.info(f"   [{config.CLASS_NAMES[idx]}]: {weight:.4f} (Samples: {class_counts[idx]})")
        
    return class_weight_dict
