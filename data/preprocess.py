import ijson
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
import logging

FILE = "part-01.json"
OUTPUT_PATH = "C:\\Users\\Tyler\\repos\\AI_Spoiler_Detection\\data\\processed"
MIN_REVIEW_LENGTH = 30
SAMPLES_PER_CLASS = 100000
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Handles loading, sampling, and splitting of IMDb spoiler dataset
    """
    
    def __init__(self, file, output_dir, random_seed=42):
        """
        Initialize preprocessor
        
        args:
            file: JSON data file
            output_dir: Directory to save processed datasets
            random_seed: Random seed for reproducibility
        """
        self.file = file
        self.output_dir = Path(output_dir)
        self.random_seed = random_seed
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, samples=None):
        """
        Load data from JSON file
        
        args:
            samples: Number of samples to load (None = load all)
            
        returns:
            DataFrame with loaded reviews
        """

        reviews = []

        # Extract reviews
        with open(self.file, 'rb') as f:
            parser = ijson.items(f, 'item')
            for i, review in enumerate(parser):
                if samples is not None and i >= samples:
                    break

                reviews.append(review)

        # Prepare df
        return pd.DataFrame(reviews)
    
    def clean_text(self, df: pd.DataFrame):
        """
        Clean review text
        
        args:
            df: DataFrame of raw reviews
            
        returns:
            DataFrame with cleaned text
        """

        # Remove extra whitespace
        df['review_detail'] = df['review_detail'].str.strip()
        df['review_summary'] = df['review_summary'].str.strip()

        # Remove HTML tags
        df['review_detail'] = df['review_detail'].apply(
            lambda x: BeautifulSoup(x, 'html.parser').get_text()
        )
        df['review_detail'] = df['review_detail'].str.replace(r'\s+', ' ', regex=True)

        # Remove very short reviews
        df = df[df['review_detail'].str.len() > MIN_REVIEW_LENGTH]

        return df
    
    def create_balanced_sample(self, df: pd.DataFrame, samples_per_class=100000):
        """
        Create balanced dataset with equal spoilers/non-spoilers
        
        args:
            df: Full DataFrame
            samples_per_class: Number of samples per class
            
        returns:
            Balanced DataFrame
        """

        # Separate spoilers and non_spoilers
        spoiler_df = df[df['spoiler_tag'] == 1]
        non_spoiler_df = df[df['spoiler_tag'] == 0]

        # Sample from each
        spoiler_df = spoiler_df.sample(n=samples_per_class, random_state=self.random_seed)
        non_spoiler_df = non_spoiler_df.sample(n=samples_per_class, random_state=self.random_seed)

        # Combine and shuffle
        balanced_df = pd.concat([spoiler_df, non_spoiler_df], axis=0)
        
        return balanced_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
    
    def split_data(self, df: pd.DataFrame, train_ratio=0.7, val_ratio=0.15):
        """
        Split data into train/validation/test sets
        
        args:
            df: DataFrame to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            
        returns:
            train_df, val_df, test_df
        """
        
        rows = len(df)
        
        # Calculate split indices
        i_train = int(rows*train_ratio)
        i_val = int(rows*(train_ratio+val_ratio))

        # Split into three dfs
        train_df = df.iloc[:i_train]
        val_df = df.iloc[i_train:i_val]
        test_df = df.iloc[i_val:]

        return train_df, val_df, test_df
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save train/val/test splits to CSV files
        
        args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
        """

        # Save summary and spoiler tag to CSV
        train_df[['review_detail', 'spoiler_tag']].to_csv(self.output_dir / 'train.csv', index=False)
        val_df[['review_detail', 'spoiler_tag']].to_csv(self.output_dir / 'val.csv', index=False)
        test_df[['review_detail', 'spoiler_tag']].to_csv(self.output_dir / 'test.csv', index=False)
        
        # Log sizes as confirmation
        logger.info(f"Samples for taining: {len(train_df)}")
        logger.info(f"Samples for validation: {len(val_df)}")
        logger.info(f"Samples for testing: {len(test_df)}")
    
    def run(self, samples=None, samples_per_class=100000):
        """
        Run full preprocessing pipeline
        
        args:
            samples: Initial samples to load (None = all)
            samples_per_class: Samples per class for balanced dataset
            
        returns:
            train_df, val_df, test_df
        """

        logger.info("Starting preprocessing pipeline...")

        # 1. Load data
        logger.info("Loading data...")
        df = self.load_data(samples=samples)
        logger.info(f"Loaded {len(df)} reviews")

        # 2. Clean text
        logger.info("Cleaning text...")
        df = self.clean_text(df)
        logger.info(f"Samples after cleaning: {len(df)}")

        # 3. Create balanced sample
        logger.info(f"Creating balanced samples ({samples_per_class} per class)...")
        df = self.create_balanced_sample(df, samples_per_class)
        logger.info(f"Balanced dataset: {len(df)} reviews")

        # 4. Split data
        logger.info("Splitting dataset into training, validation, and testing...")
        train_df, val_df, test_df = self.split_data(df, TRAIN_RATIO, VAL_RATIO)

        # 5. Save splits
        logger.info("Saving splits...")
        self.save_splits(train_df, val_df, test_df)
        logger.info("Samples saved successfully!")

def main():
    preprocessor = DataPreprocessor(file=FILE, output_dir=OUTPUT_PATH, random_seed=42)

    preprocessor.run(samples=None, samples_per_class=SAMPLES_PER_CLASS)

if __name__ == "__main__":
    main()