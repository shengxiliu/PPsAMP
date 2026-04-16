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

# use conda
conda env create -f environment.yml
conda activate ppsamp
```

## Download Pre-trained Model

Pre-trained model files are available in the [Releases](https://github.com/shengxiliu/PPsAMP/releases) section.

Required files:
- `best_model_e20p5.pth` - Trained model weights
- `feature-pseaac/smorf.h5` - Feature extraction model
- `feature-t/train.h5`, `val.h5`, `test.h5` - Dataset features

- 
## Usage

# Basic usage (assumes weights are in the root directory)
chmod +x predict.sh
./predict.sh input.fasta results.csv

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
