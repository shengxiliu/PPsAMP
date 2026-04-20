from transformers import BertModel, BertTokenizer
import re
import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import time  # 添加计时模块
import argparse # 新增：用于接收命令行参数
import os

# ================= 1. 解析命令行参数 =================
parser = argparse.ArgumentParser(description="Extract ProtBERT features for sequences.")
parser.add_argument('--fasta', type=str, required=True, help="Input sequence file path (FASTA or CSV)")
parser.add_argument('--output', type=str, required=True, help="Output feature HDF5 file path")
parser.add_argument('--model_name', type=str, default="Rostlab/prot_bert_bfd", help="HuggingFace model name or local path")
args = parser.parse_args()

# 处理模型路径：如果是 bash 脚本传来的简写，映射到完整的 HuggingFace 仓库名
model_path = args.model_name
if model_path == "prot_bert_bfd":
    model_path = "Rostlab/prot_bert_bfd"

print(f"Loading ProtBERT model from: {model_path}")

# ================= 2. 加载模型 =================
start_time = time.time()  # 记录开始时间
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
model = BertModel.from_pretrained(model_path)

# ================= 3. 读取序列 (兼容 FASTA 和 CSV) =================
sequences = []
labels = []
seq_names = []

# 判断是否是 CSV 文件
if args.fasta.lower().endswith('.csv'):
    df = pd.read_csv(args.fasta)
    sequences = df["Sequence"].tolist()
    # 如果 CSV 有标签就读取，没有就全填 0
    labels = df["Label"].tolist() if "Label" in df.columns else [0] * len(sequences)
    seq_names = sequences.copy() # 名字就用序列本身
else:
    # 按照 FASTA 格式读取
    with open(args.fasta, 'r') as file:
        first_line = file.readline().strip()
        file.seek(0)
        
        if first_line.startswith('>'):
            current_seq = ""
            for line in file:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        labels.append(0) # 推理阶段，假定标签为 0
                        seq_names.append(current_seq)
                        current_seq = ""
                else:
                    current_seq += line
            if current_seq:
                sequences.append(current_seq)
                labels.append(0)
                seq_names.append(current_seq)

# ================= 4. 提取特征 =================
BertEmbed = []

print(f"Processing {len(sequences)} sequences...")
# 使用tqdm创建进度条
for seq in tqdm(sequences, total=len(sequences)):
    # 按照 ProtBERT 的要求，给氨基酸之间加空格
    sequence_spaced = re.sub(r"(?<=\w)(?=\w)", " ", seq)
    sequence_spaced = re.sub(r"[UZOB]", "X", sequence_spaced)
    
    encoded_input = tokenizer(sequence_spaced, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
        # 获取 [CLS] token 的 embedding 或其他你需要的层
        embedding = output[1] 
        BertEmbed.append(embedding.detach().cpu().numpy())

end_time = time.time()  # 记录结束时间
total_time = end_time - start_time
average_time = total_time / len(sequences) if len(sequences) > 0 else 0
print(f"Total processing time: {total_time:.4f} seconds")
print(f"Average time per sequence: {average_time:.6f} seconds")        

# ================= 5. 保存结果 =================
BertEmbed_np = np.array([embedding[0] for embedding in BertEmbed])        
Bert_feature = pd.DataFrame(BertEmbed_np)

col = ["Bert_F" + str(i + 1) for i in range(0, Bert_feature.shape[1])]
Bert_feature.columns = col
Bert_feature.index = seq_names
Bert_feature["label"] = labels

# 确保输出目录存在
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# 将结果保存到 HDF5 文件中 (使用 'w' 模式每次覆盖，防止之前临时文件残留报错)
Bert_feature.to_hdf(args.output, key='data', mode='w', complevel=4, complib='blosc')

print(f"ProtBERT 特征已成功保存至 {args.output}")
