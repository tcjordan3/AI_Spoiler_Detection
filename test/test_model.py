from src.models.model import SpoilerClassifier
from src.data.dataset import load_data
import torch

TRAIN_FILE = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed\\train.csv"
VAL_FILE = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed\\val.csv"
TEST_FILE = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed\\test.csv"

# Load data
train_loader, _, _ = load_data(TRAIN_FILE, VAL_FILE, TEST_FILE, batch_size=8)

# Initialize model
model = SpoilerClassifier('bert-base-uncased', 2, 0.3)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Extract a batch
batch = next(iter(train_loader))

# Forward pass
with torch.no_grad():
    logits = model.forward(batch['input_ids'], batch['attention_mask'])

print(f"Logits shape: {logits.shape}")  # Should be (8, 2)
print(f"Sample logits: {logits[0]}")    # Two numbers (scores for each class)

# Determine probabilities
probs = torch.softmax(logits, dim=1)
print(f"Sample probabilities: {probs[0]}")  # Should sum to 1.0

# Predictions
preds = torch.argmax(logits, dim=1)
print(f"Predictions: {preds}")  # 0s and 1s