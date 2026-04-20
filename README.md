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
（1）git clone https://github.com/shengxiliu/PPsAMP.git
cd PPsAMP

or (2)下载上方安装包DownLoad ZIP
unzip DownLoad ZIP
cd PPsAMP-main


点击下载项目右侧release：https://github.com/shengxiliu/PPsAMP/releases
保存到当前目录命名为finetuned_protbert
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
├── data/                  # 数据集目录 (存放模型训练和评估的数据)
│   ├── test.csv           # 测试集数据
│   ├── train1.csv         # 训练集数据
│   └── val1.csv           # 验证集数据
│
├── demo/                  #  模型推理与核心代码
│   └── attention1_4.py    # 结合注意力机制执行最终预测的主程序
│
├── feature-pseaac/        # 传统特征提取模块
│   └── PseAAC_1_4.py      # 用于提取多肽序列的 PseAAC (伪氨基酸组成) 特征
│
├── feature-t/             # 深度学习特征提取模块
│   └── feature_extract_test2.py # 调用 ProtBERT 提取序列的高维语义特征
│
├── turning/               #  模型微调与训练模块 (适合进阶用户)
│   ├── finetune.sh        # 一键启动模型微调训练的脚本
│   └── training_args.bin  # 预设的训练参数配置文件
│
├── .gitattributes         # Git 配置文件 (保持不用管它)
├── README.md              # 项目说明文档 (你正在看的文件)
├── exm.fast               # 例输入文件 (提供给你的测试用 FASTA 短序列)
├── pseaac.sh              # 一键预测核心脚本 (小白必备，全自动流水线)
└── requirements.txt       #  环境依赖清单 (用于一键安装所有需要的 Python 包)

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
