import torch.nn as nn
from transformers import BertModel

class SpoilerClassifier(nn.Module):
    """
    BERT-based classifier for spoiler detection
    """
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout=0.3):
        """
        Initialize model
        
        args:
            model_name: Pretrained BERT model name
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout probability
        """

        super(SpoilerClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        returns:
            Logits for each class
        """

        # Pass through BERT
        outputs = self.bert.forward(input_ids, attention_mask)

        # Extract [CLS] token representation and apply dropout
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Pass through classification head
        logits = self.classifier(pooled_output)

        return logits