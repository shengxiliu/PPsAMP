# PPsAMP

**PPsAMP: A Deep Learning Based Framework for Identification of Short Antimicrobial Peptides by Fine-tuning Protein Language Model**



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
1.Download the relevant code
（1）git clone https://github.com/shengxiliu/PPsAMP.git
cd PPsAMP
or (2)down DownLoad ZIP
unzip DownLoad ZIP
cd PPsAMP-main

2.Download the relevant models
Click to download on the right side of the project:
release：https://github.com/shengxiliu/PPsAMP/releases

Save to the current directory and name it as:finetuned_protbert


# use conda
conda env create -f requirements.yml
conda activate ppsamp
```

- 
## Usage

# Basic usage (assumes weights are in the root directory)
chmod +x predict.sh
./predict.sh input.fasta results.csv

```bash
bash pseaac.sh exm.fast result.csv


## Project Structure

```
```text
PPsAMP/
├── data/                  
│   ├── test.csv          
│   ├── train1.csv        
│   └── val1.csv         
│
├── demo/                  
│   └── attention1_4.py   
│
├── feature-pseaac/       
│   └── PseAAC_1_4.py     
│
├── feature-t/             
│   └── feature_extract_test2.py 
│
├── turning/               
│   ├── finetune.sh      
│   └── training_args.bin 
│
├── .gitattributes         
├── README.md              
├── exm.fast              
├── pseaac.sh             
└── requirements.txt      

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
