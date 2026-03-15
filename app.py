import streamlit as st
import torch
from pathlib import Path

from src.models.model import SpoilerClassifier
from src.data.dataset import SpoilerDataset
from transformers import BertTokenizer

# Page config
st.set_page_config(
    page_title="Spoiler Detection",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    """
    Load trained model
    
    Returns:
        model: Loaded SpoilerClassifier
        tokenizer: BERT tokenizer
        device: device used for inference
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    # Load model from checkpoint
    model = SpoilerClassifier(freeze_bert=False)
    checkpoint_path = 'src/models/best_model.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    # Set to eval mode
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    return model, tokenizer, device

def predict_spoiler(text, model, tokenizer, device, max_length=512):
    """
    Predict if review contains spoilers
    
    Args:
        text: Review text
        model: Trained model
        tokenizer: BERT tokenizer
        device: torch device
        max_length: Max sequence length
        
    Returns:
        prediction: "Spoiler" or "Not Spoiler"
        confidence: Confidence score (0-1)
    """

    # Tokenize input
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Run inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    # Get prediction and confidence
    label = "Spoiler" if prediction == 1 else "Not Spoiler"
    
    return label, confidence

def main():
    # Header
    st.title("🎬 Movie Review Spoiler Detector")
    st.markdown("""
    This tool uses BERT to detect whether a movie review contains spoilers.
    Paste a review below to check!
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model, tokenizer, device = load_model()

    if 'review_text' not in st.session_state:
        st.session_state.review_text = ""
    
    # Input section
    st.subheader("Enter a movie review:")
    review_text = st.text_area(
        "Review text",
        value=st.session_state.review_text,
        placeholder="Type or paste a movie review here...",
        height=200,
        label_visibility="collapsed"
    )

    # Update session_state when text changes
    st.session_state.review_text = review_text
    
    EXAMPLES = {
        "non_spoiler": "This movie was absolutely fantastic! The acting was superb and the cinematography was beautiful. Highly recommend watching it.",
        "spoiler": "I can't believe the main character died at the end! That plot twist completely shocked me. The final battle scene was epic though."
    }

    # Example reviews (optional)
    st.markdown("**Or try an example:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Example: Non-Spoiler"):
            st.session_state.review_text = EXAMPLES["non_spoiler"]
            st.rerun()
    
    with col2:
        if st.button("Example: Spoiler"):
            st.session_state.review_text = EXAMPLES["spoiler"]
            st.rerun()
    
    # Predict button
    if st.button("Check for Spoilers", type="primary", use_container_width=True):
        if not review_text.strip():
            st.warning("Please enter a review first!")
        else:
            with st.spinner("Analyzing..."):
                # Get prediction
                prediction, confidence = predict_spoiler(review_text, model, tokenizer, device)
            
            # Display results
            st.divider()
            
            # Show results based on prediction
            if prediction == "Spoiler":
                st.error("⚠️ **Contains Spoilers!**")
            else:
                st.success("✅ **No Spoilers Detected**")
            
            # Show confidence
            st.metric("Confidence", f"{confidence:.1%}")
            
            # Show confidence bar
            st.progress(confidence)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    Built with BERT | Trained on IMDB reviews | 76% accuracy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()