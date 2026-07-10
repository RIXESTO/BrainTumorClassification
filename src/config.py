import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "BrainTumorMRI", "Training")
TEST_DIR = os.path.join(BASE_DIR, "BrainTumorMRI", "Testing")

# Output & Model Directories
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "best_brain_tumor_model.keras")

# Ensure output folders exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Dataset & Image Specs
IMG_SIZE = (224, 224)
IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
SEED = 42

# Classes
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES = len(CLASS_NAMES)

# Hyperparameters for Two-Stage Fine-Tuning
WARMUP_EPOCHS = 5        # Stage 1: Backbone frozen, train custom classification head
FINETUNE_EPOCHS = 15     # Stage 2: Unfreeze top layers of backbone, fine-tune end-to-end
LEARNING_RATE_STAGE1 = 1e-4
LEARNING_RATE_STAGE2 = 1e-5

# Backbone Choice: 'EfficientNetB0' or 'ResNet50V2'
DEFAULT_BACKBONE = 'EfficientNetB0'
