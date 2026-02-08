import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
from torch.utils.data import DataLoader
from pathlib import Path

from models.model import SpoilerClassifier

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)

class Trainer:
    """
    Handles training and evaluation of spoiler classifier
    """
    
    def __init__(self, model: SpoilerClassifier, train_loader: DataLoader, val_loader: DataLoader, device='cuda', learning_rate=2e-5, epochs=3):
        """
        Initialize trainer
        
        args:
            model: SpoilerClassifier instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        # Move model to device
        if torch.cuda.is_available():
            logger.info("Using GPU for training")
            self.device = torch.device(device)
        else: 
            logger.info("Cuda not available. Using CPU for training")
            self.device = torch.device('cpu')

        self.model.to(self.device)

        # Initialize optimizer (AdamW)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Initialize loss function (CrossEntropyLoss)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize learning rate scheduler
        total_steps = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps*0.1,  # 10% warmup
            num_training_steps=total_steps,
        )
    
    def train_epoch(self):
        """
        Train for one epoch
        
        returns:
            Average training loss for the epoch
        """

        # Set model to training mode
        self.model.train()
        total_loss = 0

        # Loop through batches
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch in progress_bar:
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model.forward(input_ids, attention_mask)

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)
    
    def evaluate(self, data_loader: DataLoader):
        """
        Evaluate model on validation/test set
        
        args:
            data_loader: DataLoader to evaluate on
            
        returns:
            average_loss: Average loss among epochs
            accuracy: Accuracy of the model
        """

        # Set model to eval mode
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        # Disable gradients
        with torch.no_grad():

            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move data to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                logits = self.model.forward(input_ids, attention_mask)

                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Get predictions
                predicted = torch.argmax(logits, dim=1)

                # Track correct predictions
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        # Return average loss and accuracy
        average_loss = total_loss / len(data_loader)
        accuracy = correct / total

        return average_loss, accuracy
    
    def train(self):
        """
        Full training loop
        
        returns:
            Dictionary with training history
        """

        best_val_acc = 0.0
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Train for one epoch
            train_loss = self.train_epoch()
            logger.info(f"Training Loss: {train_loss:.4f}")

            # Evaluate on validation set
            val_loss, val_acc = self.evaluate(self.val_loader)
            logger.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

            # Log metrics
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/best_model.pt')
                logger.info(f"Saved new best model with accuracy: {best_val_acc:.4f}")

        return history
    
    def save_model(self, path):
        """
        Save model checkpoint
        
        args:
            path: Path to save model
        """

        # Save model state_dict
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """
        Load model checkpoint
        
        args:
            path: Path to load model from
        """

        #  Load model state_dict
        self.model.load_state_dict(torch.load(path, map_location=self.device))