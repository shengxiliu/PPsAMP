# PPsAMP

**PPsAMP: A Deep Learning Based Framework for Identification of Short Antimicrobial Peptides by Fine-tuning Protein Language Model**

PPsAMP is a deep learning framework for identifying short antimicrobial peptides by fine-tuning a protein language model.

## Features

- **Multi-modal Feature Extraction**: Combines PseAAC (Pseudo Amino Acid Composition) and ProtBERT protein language model features
- **Cross-Attention Mechanism**: Uses attention mechanism to effectively integrate features from different modalities
- **Redundancy-Reduced Dataset**: Training and evaluation data are deduplicated at 40% sequence similarity to remove homologous redundancy
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

```bash
git clone https://github.com/shengxiliu/PPsAMP.git
cd PPsAMP

# download release assets: https://github.com/shengxiliu/PPsAMP/releases

# use conda
conda env create -f environment.yml
conda activate ppsamp
```

## Usage

```bash
# Basic usage (assumes weights are in the root directory)
chmod +x predict.sh
./predict.sh input.fasta results.csv

# Or use the all-in-one pipeline
bash pseaac.sh exm.fast result.csv
```

## Data Deduplication

Sequences are deduplicated by greedy clustering based on pairwise sequence similarity. Sequence lengths range from 11 to 30 amino acids.

**Algorithm** (`sequence_dedup.py`):

1. Sort sequences by length in descending order
2. For each sequence, compare to existing cluster representatives
3. Assign to the first cluster with similarity ≥ 0.40, or start a new cluster

**Similarity definition**:

```python
identity = (number of identical residues at the same position) / min(len(seq1), len(seq2))
```

No gap alignment is performed. This is a position-wise comparison that tends to underestimate true biological identity for sequences of unequal length.

**Threshold**: 0.40

## Datasets

### Data Preparation Pipeline

```
Original data (17,238)
  -> 40% similarity deduplication
7,522 representatives (2,316 AMP / 5,206 non-AMP)
  -> split into train / val / test
```

### Split Statistics

The deduplicated representatives are re-split into train / val / test at a 6:2:2 ratio.

| Dataset | Original | Deduplicated (used) |
|---|---|---|
| Train | 10,342 | **4,724** (1,514 AMP / 3,210 non-AMP) |
| Val | 3,448 | **1,398** (400 AMP / 998 non-AMP) |
| Test | 3,448 | **1,400** (402 AMP / 998 non-AMP) |
| **Total** | **17,238** | **7,522** (2,316 AMP / 5,206 non-AMP) |

## Project Structure

```
PPsAMP/
├── data/                  # dataset for training and evaluation (40% similarity deduplicated)
│   ├── train_dedup_split.csv   # training set (4,724)
│   ├── val_dedup_split.csv     # validation set (1,398)
│   └── test_dedup_split.csv    # test set (1,400)
│
├── models/                # trained model weights (40% similarity data)
│   ├── ablation_fusion.pth       # PPsAMP fusion model (main, decision threshold=0.3)
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
├── sequence_dedup.py      # data deduplication script (greedy clustering at 40% similarity)
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

## Paper-Reproduction Pipeline

The paper-aligned pipeline uses residue-level, fine-tuned ProtBert-BFD embeddings rather than a single pooled sequence vector. It is separate from the legacy global-feature checkpoints in `models/`.

```
data/*.csv
  -> fine-tuned ProtBert-BFD token embeddings [B, L, 1024] + padding masks
  -> PseAAC descriptors [B, 420]
  -> PseAAC as Query; ProtBert tokens as Key/Value
  -> Cross-Attention + two Self-Attention/FFN blocks
  -> binary AMP classifier
```

### Included trained assets

- `models/protbert_finetuned/`: fine-tuned ProtBert-BFD checkpoint used for token-level feature extraction (Git LFS).
- `models/token_attention/best_model.pt`: trained token-level Cross-Attention classifier. This checkpoint produced the paper's best single-run test result: Accuracy 0.8700, F1 0.7708, MCC 0.6802, Sensitivity 0.7612, and Specificity 0.9138.

Fetch LFS assets after cloning:

```bash
git lfs install
git lfs pull
```

### Run the complete workflow

The following command extracts token-level ProtBert and PseAAC features from `data/`, trains the fusion classifier, selects the checkpoint using validation F1, and evaluates it on the held-out test set:

```bash
bash pipeline/run_paper_pipeline.sh
```

The classifier uses PseAAC as the Cross-Attention Query and masked per-residue ProtBert embeddings as Key and Value. No attention-weight visualization scripts or visualization outputs are part of this workflow.
