import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import os

# ==================== 1. 模型架构定义 ====================
class CrossAttentionModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, pseaac_dim, protbert_dim, num_classes, dropout=0.5):
        super(CrossAttentionModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pseaac_projection = nn.Linear(pseaac_dim, embedding_dim)
        self.protbert_projection = nn.Linear(protbert_dim, embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, pseaac_features, protbert_features):
        pseaac_features = self.pseaac_projection(pseaac_features)
        protbert_features = self.protbert_projection(protbert_features)
        pseaac_features = pseaac_features.unsqueeze(0)
        protbert_features = protbert_features.unsqueeze(0)
        attn_output, _ = self.cross_attention(query=pseaac_features, key=protbert_features, value=protbert_features)
        for _ in range(self.num_layers):
            self_attn_output, _ = self.self_attention(query=attn_output, key=attn_output, value=attn_output)
            attn_output = self.layer_norm1(attn_output + self_attn_output)
            ff_output = self.feed_forward(attn_output)
            attn_output = self.layer_norm2(attn_output + ff_output)
        attn_output = attn_output.squeeze(0)
        logits = self.classifier(attn_output)
        return logits


# ==================== 2. 主程序入口 ====================
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run inference using trained Cross-Attention model.")
    parser.add_argument('--pseaac_input', type=str, required=True, help="Path to PseAAC features CSV")
    parser.add_argument('--bert_input', type=str, required=True, help="Path to ProtBERT features HDF5")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained .pth model weights")
    parser.add_argument('--output', type=str, required=True, help="Path to save the final prediction CSV")
    args = parser.parse_args()

    print("[Info] Loading features...")
    
    # 1. 加载 PseAAC 特征
    df_pseaac = pd.read_csv(args.pseaac_input, header=None)
    pseaac_features = torch.tensor(df_pseaac.values, dtype=torch.float32)

    # 2. 加载 ProtBERT 特征
    df_bert = pd.read_hdf(args.bert_input)
    sequences = df_bert.index.tolist()  # 提取序列名称
    # 丢弃 label 列，只保留特征
    if 'label' in df_bert.columns:
        df_bert = df_bert.drop(columns=['label'])
    protbert_features = torch.tensor(df_bert.values, dtype=torch.float32)

    assert len(pseaac_features) == len(protbert_features), "PseAAC and ProtBERT feature counts do not match!"

    # 构建 DataLoader
    batch_size = 32
    test_dataset = TensorDataset(pseaac_features, protbert_features)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. 动态获取特征维度 & 模型初始化
    pseaac_dim = pseaac_features.shape[1]
    protbert_dim = protbert_features.shape[1]
    
    embedding_dim = 256
    num_heads = 4
    num_layers = 2
    num_classes = 2
    dropout = 0.5

    device = torch.device("cpu")
    print(f"[Info] Using device: {device}")
    
    model = CrossAttentionModel(embedding_dim, num_heads, num_layers, pseaac_dim, protbert_dim, num_classes, dropout)
    
    # 加载权重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except Exception as e:
        print(f"[Error] Failed to load model weights from {args.model_path}. Please check the file.")
        raise e
        
    model.to(device)
    model.eval()

    # 4. 运行预测
    results = []

    with torch.no_grad():
        for batch_idx, (pseaac_batch, protbert_batch) in enumerate(tqdm(test_loader, desc="Testing")):
            pseaac_batch, protbert_batch = pseaac_batch.to(device), protbert_batch.to(device)
            outputs = model(pseaac_batch, protbert_batch)
            
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()

            start_idx = batch_idx * batch_size
            end_idx = start_idx + len(predictions)
            batch_sequences = sequences[start_idx:end_idx]

            for i in range(len(predictions)):
                results.append({
                    "Sequence": batch_sequences[i],
                    "Prediction": "Positive (AMP)" if predictions[i] == 1 else "Negative (Non-AMP)",
                    "Prob_Negative": round(probabilities[i][0], 4),
                    "Prob_Positive": round(probabilities[i][1], 4)
                })

    # 5. 保存结果
    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results_df.to_csv(args.output, index=False)

    print(f"\n✅ Prediction complete! Results successfully saved to: {args.output}")
