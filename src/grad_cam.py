import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import src.config as config
from src.utils import logger, check_device

class GradCAM:
    """
    Medical XAI Engine: Gradient-weighted Class Activation Mapping (Grad-CAM).
    Produces visual heatmaps highlighting exact tumor regions that influenced
    the deep learning model's diagnosis.
    """
    def __init__(self, model, layer_name=None):
        self.model = model
        self.backbone = self._find_backbone(model)
        self.layer_name = layer_name or self._find_target_conv_layer(self.backbone)
        logger.info(f"🔍 Grad-CAM initialized with target conv layer: '{self.layer_name}' inside backbone '{self.backbone.name}'")
        
        # Build Grad-CAM gradient extraction sub-model
        # We need output of target conv layer AND output of full model
        self.grad_model = self._build_grad_model()

    def _find_backbone(self, model):
        """Locate the transfer learning backbone model layer, ignoring preprocessing/augmentation Sequentials."""
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and not isinstance(layer, tf.keras.Sequential):
                return layer
        return model

    def _find_target_conv_layer(self, backbone):
        """Automatically find the last 4D convolutional layer in the backbone."""
        # Check standard layer names first
        for layer in reversed(backbone.layers):
            if layer.name in ['top_conv', 'post_relu', 'conv5_block3_out']:
                return layer.name
        
        # Check by layer type and shape
        for layer in reversed(backbone.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                shape = getattr(layer, 'output_shape', None)
                if shape is not None and len(shape) == 4:
                    return layer.name
                # If output_shape attribute is None or uncomputed in Keras 3, fallback to checking layer name/type
                return layer.name
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")

    def _build_grad_model(self):
        """Construct model mapping from raw image input to conv features and final logits."""
        # If backbone is the main model itself or target conv is directly inside model.layers
        if self.backbone == self.model or self.layer_name in [l.name for l in self.model.layers]:
            target_conv_layer = self.model.get_layer(self.layer_name)
            return tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_conv_layer.output, self.model.output]
            )
        
        # Otherwise if backbone is a nested functional sub-model
        target_conv_layer = self.backbone.get_layer(self.layer_name)
        backbone_feature_extractor = tf.keras.Model(
            inputs=self.backbone.inputs,
            outputs=target_conv_layer.output
        )
        return tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[
                backbone_feature_extractor(self.model.get_layer(self.backbone.name).input),
                self.model.output
            ]
        )

    def compute_heatmap(self, img_array, class_index=None):
        """
        Compute Grad-CAM heatmap for a preprocessed image tensor (1, 224, 224, 3).
        If class_index is None, computes explanation for the top predicted class.
        """
        # Ensure 4D batch tensor
        if len(img_array.shape) == 3:
            img_array = np.expand_dims(img_array, axis=0)
        img_tensor = tf.cast(img_array, tf.float32)

        # Apply backbone-specific preprocessing scaling
        if config.DEFAULT_BACKBONE == 'Xception':
            img_tensor = tf.keras.applications.xception.preprocess_input(img_tensor)
        elif config.DEFAULT_BACKBONE == 'ResNet50V2':
            img_tensor = tf.keras.applications.resnet_v2.preprocess_input(img_tensor)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor, training=False)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        # Extract gradients of target class probability wrt conv feature map
        grads = tape.gradient(loss, conv_outputs)
        
        if grads is None:
            logger.warning(f"⚠️ Could not compute gradients for target conv layer '{self.layer_name}'. Returning zero heatmap.")
            return np.zeros((config.IMG_SIZE[0], config.IMG_SIZE[1]), dtype=np.float32), int(class_index), predictions[0].numpy()

        # Pool gradients over spatial dimensions (height and width) to get channel weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in feature map by its importance weight
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        # Squeeze only the trailing channel dimension to preserve (H, W) even if H=1 or W=1
        heatmap = tf.squeeze(heatmap, axis=-1)

        # Apply ReLU to keep only features that have positive influence on the target class
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize heatmap to [0.0, 1.0]
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap /= max_val
            
        return heatmap.numpy(), int(class_index), predictions[0].numpy()

    def generate_visual_explanation(self, img_path, alpha=0.45):
        """
        Load an MRI scan from path, generate Grad-CAM heatmap, overlay jet colormap,
        and return (original_img, heatmap_colored, superimposed_img, pred_class, pred_probs).
        """
        # Load and resize image to exact network dimensions
        orig_img = Image.open(img_path).convert('RGB')
        orig_img = orig_img.resize(config.IMG_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(orig_img)

        # Compute heatmap
        heatmap, pred_idx, probs = self.compute_heatmap(img_array)

        # Resize heatmap to match image dimensions (224x224)
        heatmap_resized = cv2.resize(heatmap, (config.IMG_SIZE[0], config.IMG_SIZE[1]))
        
        # Convert to 8-bit colormap (Jet / Rainbow)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Superimpose heatmap onto original MRI scan safely preventing overflow
        superimposed_float = img_array.astype(np.float32) * (1.0 - alpha) + heatmap_colored.astype(np.float32) * alpha
        superimposed_img = np.clip(superimposed_float, 0, 255).astype(np.uint8)

        return img_array, heatmap_colored, superimposed_img, pred_idx, probs

def generate_sample_explanations(model_path=None, num_samples_per_class=1):
    """
    Generate and save a multi-class XAI diagnostic grid of MRI scans vs Grad-CAM heatmaps.
    Supports generating multiple sample rows per class via num_samples_per_class.
    """
    check_device()
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH

    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at: {model_path}. Run `python main.py --train` first.")
        return

    logger.info(f"📂 Loading model for Grad-CAM explanations from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    grad_cam = GradCAM(model)

    total_rows = config.NUM_CLASSES * max(1, num_samples_per_class)
    fig, axes = plt.subplots(total_rows, 3, figsize=(13, 4.3 * total_rows))
    if total_rows == 1:
        axes = [axes]

    logger.info(f"🔥 Generating Grad-CAM heatmaps across all tumor classes ({num_samples_per_class} sample(s)/class)...")
    
    current_row = 0
    for cls_idx, cls_name in enumerate(config.CLASS_NAMES):
        cls_folder = os.path.join(config.TEST_DIR, cls_name)
        if not os.path.exists(cls_folder):
            continue
        images = sorted([f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not images:
            continue
        
        # Pick up to num_samples_per_class images
        selected_images = images[:num_samples_per_class]
        for img_name in selected_images:
            sample_path = os.path.join(cls_folder, img_name)
            orig_img, _, superimposed_img, pred_idx, probs = grad_cam.generate_visual_explanation(sample_path)
            pred_name = config.CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            # Column 1: Original MRI
            axes[current_row, 0].imshow(orig_img)
            axes[current_row, 0].set_title(f"Ground Truth: {cls_name}\n({img_name})", fontsize=11, fontweight='bold')
            axes[current_row, 0].axis('off')

            # Column 2: Grad-CAM Heatmap Superimposed
            axes[current_row, 1].imshow(superimposed_img)
            title_color = "darkgreen" if pred_idx == cls_idx else "darkred"
            axes[current_row, 1].set_title(f"Grad-CAM (Pred: {pred_name} | {confidence:.1f}%)", fontsize=11, fontweight='bold', color=title_color)
            axes[current_row, 1].axis('off')

            # Column 3: Class Probability Bar Chart
            y_pos = np.arange(config.NUM_CLASSES)
            bars = axes[current_row, 2].barh(y_pos, probs * 100, color='#3498db', alpha=0.85)
            axes[current_row, 2].set_yticks(y_pos)
            axes[current_row, 2].set_yticklabels(config.CLASS_NAMES, fontsize=10)
            axes[current_row, 2].invert_yaxis()
            axes[current_row, 2].set_xlim([0, 105])
            axes[current_row, 2].set_xlabel("Confidence (%)", fontsize=10)
            axes[current_row, 2].set_title("Softmax Probability Output", fontsize=11)
            
            # Highlight predicted bar
            bars[pred_idx].set_color('#2ecc71' if pred_idx == cls_idx else '#e74c3c')
            for bar in bars:
                width = bar.get_width()
                axes[current_row, 2].text(width + 2, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontsize=9)
            
            current_row += 1

    plt.tight_layout()
    output_path = os.path.join(config.OUTPUTS_DIR, "grad_cam_samples.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"🖼️ Saved diagnostic Grad-CAM grid to: {output_path}")

if __name__ == "__main__":
    generate_sample_explanations()
