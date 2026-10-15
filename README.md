# PPsAMP

**PPsAMP: A Deep Learning Based Framework for Identification of Short Antimicrobial Peptides by Fine-tuning Protein Language Model**

PPsAMP is a deep learning framework for identifying short antimicrobial peptides by fine-tuning a protein language model.

## Features

- **Multi-modal Feature Extraction**: Combines PseAAC (Pseudo Amino Acid Composition) and ProtBERT protein language model features
- **Cross-Attention Mechanism**: Uses attention mechanism to effectively integrate features from different modalities
- **Redundancy-Reduced Dataset**: Training and evaluation data are deduplicated by CD-HIT at 40% sequence identity to remove homologous redundancy
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

# Download
```bash
git clone https://github.com/shengxiliu/PPsAMP.git
cd PPsAMP

# download release assets: https://github.com/shengxiliu/PPsAMP/releases

# use conda
conda env create -f environment.yml
conda activate ppsamp
```

- 
## Usage

# Basic usage (assumes weights are in the root directory)
chmod +x predict.sh
./predict.sh input.fasta results.csv

```bash
bash pseaac.sh exm.fast result.csv

## Datasets

All sequences are deduplicated by **CD-HIT at 40% sequence identity** to eliminate homologous redundancy. Sequence lengths range from 11 to 30 amino acids.

### Data Preparation Pipeline

```
Original data (17,238)
  → CD-HIT 40% redundancy removal
CD-HIT 40% representatives (7,522: 2,316 AMP / 5,206 non-AMP)
  → split into train / val / test
```

### Split Statistics

The CD-HIT 40% representatives are re-split into train / val / test at a 6:2:2 ratio.

| Dataset | Original | CD-HIT 40% (used) |
|---|---|---|
| Train | 10,342 | **4,724** (1,514 AMP / 3,210 non-AMP) |
| Val | 3,448 | **1,398** (400 AMP / 998 non-AMP) |
| Test | 3,448 | **1,400** (402 AMP / 998 non-AMP) |
| **Total** | **17,238** | **7,522** (2,316 AMP / 5,206 non-AMP) |

## Project Structure

```
```text
PPsAMP/
├── data/                  # dataset for training and evaluation (CD-HIT 40% deduplicated)
│   ├── train_cdhit40_split.csv   # training set (4,724)
│   ├── val_cdhit40_split.csv     # validation set (1,398)
│   ├── test_cdhit40_split.csv    # test set (1,400)
│   └── test_cdhit40_split.fasta  # test set in FASTA format
│
├── models/                # trained model weights (CD-HIT 40% data)
│   ├── ablation_fusion.pth       # PPsAMP fusion model (main, threshold=0.3)
│   ├── ablation_prot_only.pth    # ProtBERT-only model
│   └── ablation_pse_only.pth     # PseAAC-only model
│
├── demo/                  # inference and core code
│   └── attention1_4.py    # main program for final prediction with attention mechanism
│
├── feature-pseaac/        # conventional feature extraction module
│   └── PseAAC_1_4.py      # extract PseAAC (pseudo amino acid composition) features
│
├── feature-t/             # deep learning feature extraction module
│   └── feature_extract_test2.py # extract high-dimensional semantic features with ProtBERT
│
├── turning/               # fine-tuning and training module (for advanced users)
│   ├── finetune.sh        # one-click script to start fine-tuning
│   └── training_args.bin  # preset training argument configuration
│
├── .gitattributes         # Git configuration file (no need to modify)
├── README.md              # project documentation (the file you are reading)
├── exm.fast               # example input file (short FASTA sequences for testing)
├── pseaac.sh              # one-click prediction script (fully automated pipeline)
└── requirements.txt       # environment dependency list (install all required Python packages)

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
