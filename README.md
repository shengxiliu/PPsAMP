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

# Download
```bash
git clone [https://github.com/shengxiliu/PPsAMP.git](https://github.com/shengxiliu/PPsAMP.git)
cd PPsAMP

# use conda
conda env create -f environment.yml
conda activate ppsamp
```

- 
## Usage

# Basic usage (assumes weights are in the root directory)
chmod +x predict.sh
./predict.sh input.fasta results.csv

## Project Structure

```


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
