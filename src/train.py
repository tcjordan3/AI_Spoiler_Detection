import logging
from data.dataset import load_data
from models.model import SpoilerClassifier
from models.trainer import Trainer
from pathlib import Path
import os

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
EPOCHS = 5
DEVICE = 'cuda'

# Data paths
if 'COLAB_GPU' in os.environ or os.path.exists('/content'):
    BASE_DIR = Path('/content/drive/MyDrive/AI_Spoiler_Detection')
else:
    BASE_DIR = Path(__file__).parent.parent

TRAIN_PATH = BASE_DIR / 'data' / 'processed' / 'train.csv'
VAL_PATH = BASE_DIR / 'data' / 'processed' / 'val.csv'
TEST_PATH = BASE_DIR / 'data' / 'processed' / 'test.csv'

def main():
    logger.info(f"BATCH SIZE: {BATCH_SIZE}, LEARNING RATE: {LEARNING_RATE}, EPOCHS: {EPOCHS}")
    
    # 1. Load data
    train_loader, val_loader, test_loader = load_data(TRAIN_PATH, VAL_PATH, TEST_PATH, BATCH_SIZE)

    # 2. Initialize model
    model = SpoilerClassifier(freeze_bert=False)

    # After model initialization
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # 3. Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE, learning_rate=LEARNING_RATE, epochs=EPOCHS)

    # 4. Train
    trainer.train()
    trainer.load_model('models/best_model.pt')

    # 5. Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)

    # 6. Log final results
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
