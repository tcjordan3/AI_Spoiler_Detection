# AI_Spoiler_Detection

Fine-tuned BERT model for spoiler detection in film reviews. Trained on IMDb Review dataset. For more details on the model please visit [bert-base-spoiler-detection](https://huggingface.co/tcjordan3/bert-base-spoiler-detection)

<img width="918" height="850" alt="Example_Spoiler" src="https://github.com/user-attachments/assets/ffe33528-ff17-415a-a3fd-6266542bfc8a" />

## Dataset

**Source:** [IMDB Review Dataset](https://www.kaggle.com/datasets/ebiswas/imdb-review-dataset) by Enam Biswas (2021)
```bibtex
@misc{enam biswas_2021, 
  title={IMDb Review Dataset - ebD}, 
  url={https://www.kaggle.com/dsv/1836923}, 
  DOI={10.34740/KAGGLE/DSV/1836923}, 
  publisher={Kaggle}, 
  author={Enam Biswas}, 
  year={2021}
}
```

## Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11
- (Optional) CUDA-compatible GPU for faster inference

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/tcjordan3/AI_Spoiler_Detection.git
cd AI_Spoiler_Detection
```

2. **Create a virtual environment**

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\\Scripts\\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install PyTorch with CUDA support** (for GPU inference)

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only (no GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4. **Install the package**
```bash
pip install -e .
```

### Running the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

On first run, the trained model (~440MB) will be downloaded automatically from Hugging Face. This may take a few minutes depending on your internet connection.
