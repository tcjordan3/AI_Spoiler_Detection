import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer

class SpoilerDataset(Dataset):
    """
    PyTorch Dataset for spoiler detection.
    Handles loading CSV and tokenizing text for BERT
    """
    
    def __init__(self, csv_path, tokenizer: BertTokenizer, max_length=512):
        """
        Initialize dataset
        
        args:
            csv_path: Path to CSV file (train.csv, val.csv, or test.csv)
            tokenizer: BERT tokenizer instance
            max_length: Maximum sequence length (BERT max is 512)
        """

        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        """
        Return total number of samples
        
        returns:
            Number of reviews in dataset
        """
        
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        args:
            idx: Index of sample to retrieve
            
        returns:
            Dictionary with:
                - input_ids: Tokenized text as tensor
                - attention_mask: Mask for padding as tensor
                - label: Spoiler tag (0 or 1) as tensor
        """

        # Get review text and label from DataFrame at idx
        review = self.df.iloc[idx]['review_detail']
        label = self.df.iloc[idx]['spoiler_tag']

        # Tokenize the text using self.tokenizer
        encoding = self.tokenizer(
            review,
            max_length = self.max_length,
            padding = 'max_length',  # Pad to max length
            truncation = True,      # Truncate if too long
            return_tensors = 'pt',  # Return PyTorch tensors
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(train_path, val_path, test_path, batch_size=16):
    """
    Load train/val/test datasets and create DataLoaders
    
    args:
        train_path: Path to train.csv
        val_path: Path to val.csv
        test_path: Path to test.csv
        batch_size: Batch size for training
        
    returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create three SpoilerDataset instances
    train_dataset = SpoilerDataset(train_path, tokenizer)
    val_dataset = SpoilerDataset(val_path, tokenizer)
    test_dataset = SpoilerDataset(test_path, tokenizer)

    # Wrap each in a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader