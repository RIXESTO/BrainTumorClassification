import tensorflow as tf
from src.dataset import get_augmentation_layer
from src.utils import logger

def build_transfer_model(config, backbone_type=None):
    """
    Build a production-grade transfer learning model (default EfficientNetB0)
    with integrated data augmentation and a robust multi-class classification head.
    """
    if backbone_type is None:
        backbone_type = config.DEFAULT_BACKBONE

    logger.info(f"🏗️ Building Transfer Learning Architecture using: {backbone_type}")
    
    # Input tensor
    inputs = tf.keras.layers.Input(shape=config.IMG_SHAPE, name="input_image")
    
    # Integrate domain-safe data augmentations right inside the model graph
    # (Dynamically applied during training, automatically bypassed during inference/testing)
    augmentation_layer = get_augmentation_layer()
    x = augmentation_layer(inputs)
    
    # Initialize Backbone
    if backbone_type == 'EfficientNetB0':
        # EfficientNetB0 expects pixel inputs in [0, 255] and handles internal rescaling
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
    elif backbone_type == 'ResNet50V2':
        # ResNet50V2 expects [-1, 1] scaling
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        backbone = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights='imagenet',
            input_tensor=x
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    # Freeze backbone initially for Stage 1 (Warmup)
    backbone.trainable = False
    
    # Custom Medical Classification Head
    features = backbone.output
    pooled = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    bn = tf.keras.layers.BatchNormalization(name="head_bn")(pooled)
    drop1 = tf.keras.layers.Dropout(0.5, name="head_dropout_1")(bn)
    dense1 = tf.keras.layers.Dense(256, activation="relu", name="head_dense_256")(drop1)
    drop2 = tf.keras.layers.Dropout(0.3, name="head_dropout_2")(dense1)
    outputs = tf.keras.layers.Dense(
        config.NUM_CLASSES,
        activation="softmax",
        dtype="float32",
        name="predictions"
    )(drop2)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"BrainTumor_{backbone_type}")
    
    # Compile for Stage 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE_STAGE1),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    
    logger.info(f"✅ Successfully compiled Stage 1 model (Backbone Frozen). Total Params: {model.count_params():,}")
    return model, backbone

def unfreeze_backbone_for_finetuning(model, backbone, num_layers_to_unfreeze=30, learning_rate=1e-5):
    """
    Unfreeze the top layers of the backbone for Stage 2 fine-tuning while
    strictly keeping BatchNormalization layers frozen to protect learned statistics.
    """
    logger.info(f"🔓 Unfreezing top {num_layers_to_unfreeze} layers of backbone for Stage 2 fine-tuning...")
    backbone.trainable = True
    
    total_layers = len(backbone.layers)
    for layer in backbone.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False
        
    # Keep all BatchNormalization layers frozen during fine-tuning (Critical best practice)
    bn_frozen_count = 0
    for layer in backbone.layers[-num_layers_to_unfreeze:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            bn_frozen_count += 1
            
    # Re-compile model with Stage 2 lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )
    
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    logger.info(f"✅ Stage 2 Re-compiled. Trainable layers: {num_layers_to_unfreeze} (BN frozen: {bn_frozen_count}). Trainable params: {trainable_params:,}")
    return model
