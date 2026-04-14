# PPsAMP

**PPsAMP: A Deep Learning Based Framework for Identification of Short Antimicrobial Peptides by Fine-tuning Protein Language Model**

PPsAMP 是一个基于深度学习框架，通过微调蛋白质语言模型来识别短链抗菌肽的工具。

## Features

- **Multi-modal Feature Extraction**: Combines PseAAC (Pseudo Amino Acid Composition) and ProtBERT protein language model features
- **Cross-Attention Mechanism**: Uses attention mechanism to effectively integrate features from different modalities
- **High Accuracy**: Achieves excellent performance on benchmark datasets

## Installation

### Environment Requirements

- Python >= 3.8
- PyTorch >= 1.12
- Transformers >= 4.46
- scikit-learn >= 0.23
- Biopython >= 1.83
- pandas, numpy, h5py

### Install Dependencies

use pip:

```bash
pip install -r requirements.txt
```

## Usage

### Feature Extraction

#### 1. PseAAC Feature Extraction

```bash
python feature-pseaac/PseAAC_1_4.py --fasta input.fasta --output features.csv
```

#### 2. ProtBERT Feature Extraction

```bash
python feature-t/feature_extract_test2.py --fasta input.fasta --model_name prot_bert_bfd --output features.h5
```

### Model Training

```bash
cd turning
bash finetune.sh
```

Training parameters:
- Learning rate: 2e-5
- Batch size: 16 (train) / 32 (eval)
- Epochs: 110
- Max sequence length: 30
- FP16 training enabled

### Prediction

```python
import torch
import pandas as pd
from attention1_4 import AttentionModel

# Load features
pseaac_features = pd.read_csv("pseaac_features.csv")
protbert_features = pd.read_hdf("protbert_features.h5")

# Initialize model
model = AttentionModel(pseaac_dim=420, protbert_dim=1024)
model.load_state_dict(torch.load("best_model_e20p5.pth"))

# Predict
model.eval()
with torch.no_grad():
    outputs = model(pseaac_features, protbert_features)
    predictions = torch.argmax(outputs, dim=1)
```

## Project Structure

```
PPsAMP/
├── data/                    # Dataset files
│   ├── train1.csv          # Training data
│   ├── val1.csv            # Validation data
│   └── test.csv            # Test data
├── demo/                    # Demo and visualization
│   └── attention1_4.py     # Prediction script
├── feature-pseaac/          # PseAAC feature extraction
│   └── PseAAC_1_4.py
├── feature-t/               # ProtBERT feature extraction
│   └── feature_extract_test2.py
├── turning/                 # Model training
│   ├── finetune.sh         # Training script
│   └── training_args.bin   # Training arguments
├── requirements.txt         # Python dependencies
├── environment.yml           # Conda environment
└── README.md
```

## Model Architecture

The model uses a Cross-Attention architecture:

1. **Feature Input**:
   - PseAAC features (420 dimensions)
   - ProtBERT protein language model features (1024 dimensions)

2. **Projection Layers**: Project both features to a common embedding space (256 dimensions)

3. **Attention Layers**:
   - Cross-Attention: Interacts between PseAAC and ProtBERT features
   - Self-Attention: Captures internal relationships

4. **Classification Head**: Outputs binary classification (AMP vs non-AMP)

## Performance

Model performance on test set:
- Accuracy: ~95%
- F1 Score: ~0.94

## Download Pre-trained Model

Pre-trained model files are available in the [Releases](https://github.com/shengxiliu/PPsAMP/releases) section.

Required files:
- `best_model_e20p5.pth` - Trained model weights
- `feature-pseaac/smorf.h5` - Feature extraction model
- `feature-t/train.h5`, `val.h5`, `test.h5` - Dataset features

## Citation

If you use PPsAMP in your research, please cite:

```
@article{PPsAMP,
  title={PPsAMP: A Deep Learning Based Framework for Identification of Short Antimicrobial Peptides},
  author={},
  year={2024}
}
```

## Contact

For questions and issues, please open an issue on GitHub.
