from src.models.dataset import load_data

TRAIN_FILE = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed\\train.csv"
VAL_FILE = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed\\val.csv"
TEST_FILE = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed\\test.csv"

train_loader, val_loader, test_loader = load_data(
    TRAIN_FILE, VAL_FILE, TEST_FILE, batch_size=8
)

# Extract a batch
batch = next(iter(train_loader))
print("Batch keys:", batch.keys())
print("Input IDs shape:", batch['input_ids'].shape)  # Should be (8, 512)
print("Attention mask shape:", batch['attention_mask'].shape)  # Should be (8, 512)
print("Labels shape:", batch['label'].shape)  # Should be (8,)
print("Sample label:", batch['label'][0].item())  # Should be 0 or 1