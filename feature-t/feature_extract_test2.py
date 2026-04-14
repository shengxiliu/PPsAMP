from transformers import BertModel, BertTokenizer
import re
import torch
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import time  # 添加计时模块

start_time = time.time()  # 记录开始时间
tokenizer = BertTokenizer.from_pretrained('/home/zhaozhimiao/XZ/HLAB/model_finetune/newnewnew_p3e110(2)', do_lower_case=False )
model = BertModel.from_pretrained("/home/zhaozhimiao/XZ/HLAB/model_finetune/newnewnew_p3e110(2)")
BertEmbed = []
SeqName = []
labels = []
df = pd.read_csv("/home/zhaozhimiao/XZ/HLAB/dataset/test.csv")
# # 使用tqdm创建进度条
for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    sequence = re.sub(r"(?<=\w)(?=\w)", " ", row["Sequence"])
    sequence = re.sub(r"[UZOB]", "X", sequence)
    encoded_input = tokenizer(sequence, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
        embedding = output[1]
        BertEmbed.append(embedding.detach().cpu().numpy())
        SeqName.append(sequence)
        labels.append(row["Label"])

end_time = time.time()  # 记录结束时间
total_time = end_time - start_time
average_time = total_time / df.shape[0]
print(f"Total processing time: {total_time:.4f} seconds")
print(f"Average time per sequence: {average_time:.6f} seconds")        
BertEmbed_np = np.array([embedding[0] for embedding in BertEmbed])        
Bert_feature = pd.DataFrame(BertEmbed_np)
col = ["Bert_F" + str(i + 1) for i in range(0, Bert_feature.shape[1])]
Bert_feature.columns = col
Bert_feature.index = SeqName
Bert_feature["label"] = labels
# 将结果保存到 HDF5 文件中
Bert_feature.to_hdf('/home/zhaozhimiao/XZ/HLAB/features/newnewnew_data/p3e110(2)/new/test.h5', key='data', mode='a', complevel=4, complib='blosc', append=True)
