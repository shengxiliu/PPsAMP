#!/usr/bin/env python3
"""
第一步：特征提取
  1. 用微调后的 ProtBERT 提取 token-level 特征 (last_hidden_state) → .h5
  2. 计算 PseAAC 420 维特征 → .csv

输入: train_cdhit40_split.csv / val_cdhit40_split.csv / test_cdhit40_split.csv
输出: features/train_tokens.h5, features/train_pseaac.csv, ...
"""

import os, sys, argparse, time, re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from pathlib import Path
from tqdm import tqdm

# ===================== PseAAC 计算 =====================

# 20 种标准氨基酸
AA_ORDER = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# 20 种理化性质 (来自 AAindex 数据库)
# 每行对应一个氨基酸 (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)
AA_PROPERTIES = np.array([
    # p1: Hydrophobicity (Kyte-Doolittle)
    [ 1.8,  2.5, -3.5, -3.5,  2.8, -0.4, -3.2,  4.5, -3.9,  3.8,
      1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7,  4.2, -0.9, -1.3],
    # p2: Hydrophilicity (Hopp-Woods)
    [-0.5, -1.0,  3.0,  3.0, -2.5,  0.0, -0.5, -1.8,  3.0, -1.8,
     -1.3,  0.2,  0.0,  0.2,  3.0,  0.3, -0.4, -1.5, -3.4, -2.3],
    # p3: Side chain mass
    [ 15.0, 47.0, 59.0, 73.0, 91.0,  1.0, 82.0, 57.0, 73.0, 57.0,
     75.0, 58.0, 42.0, 72.0,101.0, 31.0, 45.0, 43.0,130.0,107.0],
    # p4: pK1 (alpha-COOH)
    [2.34, 1.96, 2.09, 2.19, 1.83, 2.34, 1.82, 2.36, 2.18, 2.36,
     2.28, 2.02, 1.99, 2.17, 2.17, 2.21, 2.11, 2.32, 2.38, 2.20],
    # p5: pK2 (alpha-NH3+)
    [9.69, 10.28, 9.66, 9.67, 9.13, 9.60, 9.17, 9.68, 8.95, 9.60,
     9.21, 8.80, 10.60, 9.13, 9.04, 9.15, 9.62, 9.62, 9.39, 9.11],
    # p6: pI (isoelectric point)
    [6.00, 5.07, 2.77, 3.22, 5.48, 5.97, 7.59, 6.02, 9.74, 5.98,
     5.74, 5.41, 6.30, 5.65, 10.76, 5.68, 5.66, 5.96, 5.89, 5.66],
    # p7: Polarity (Grantham)
    [ 8.1,  5.5, 13.0, 12.3,  5.2,  9.0, 10.4,  5.2, 11.3,  4.9,
      5.7, 11.6,  8.0, 10.5, 10.5,  9.2,  8.6,  5.9,  5.4,  6.2],
    # p8: Polarizability
    [0.046, 0.128, 0.105, 0.151, 0.290, 0.000, 0.230, 0.186, 0.219, 0.186,
     0.221, 0.134, 0.131, 0.180, 0.291, 0.062, 0.108, 0.140, 0.409, 0.298],
    # p9: Solvent accessibility (Janin)
    [ 3.4,  2.2, 10.4,  9.0,  2.8,  6.2,  8.3,  1.2, 12.5,  1.6,
      3.2,  8.8,  5.0,  8.0, 13.0,  6.2,  6.4,  1.8,  1.6,  7.0],
    # p10: Normalized van der Waals volume
    [ 1.00, 2.43, 2.78, 3.78, 5.89, 0.00, 4.66, 4.00, 4.77, 4.00,
      4.43, 2.95, 2.72, 3.95, 6.13, 1.60, 2.60, 3.00, 8.08, 6.47],
    # p11: Flexibility (Bhaskaran)
    [0.357, 0.346, 0.511, 0.497, 0.314, 0.544, 0.323, 0.462, 0.466, 0.365,
     0.295, 0.411, 0.509, 0.493, 0.529, 0.561, 0.444, 0.386, 0.305, 0.420],
    # p12: Buried residues (Janin)
    [10.2,  4.8,  3.9,  4.1,  4.4,  6.2,  3.3,  7.7,  1.0,  8.9,
      4.0,  2.5,  3.9,  2.7,  1.4,  5.4,  5.3,  7.5,  3.8,  2.0],
    # p13: Average accessible surface area (Rose)
    [118.3, 146.1, 158.1, 186.3, 227.9, 85.2, 195.4, 177.8, 208.4, 177.8,
     200.5, 162.1, 147.1, 179.8, 253.3, 117.4, 145.1, 158.2, 276.9, 245.3],
    # p14: Transfer free energy (Janin)
    [ 2.6,  3.7, -4.5, -3.9,  2.0,  0.0, -3.1,  5.0, -6.5,  4.7,
      2.8,  2.8, -1.2, -2.5, -7.5, -0.5,  0.6,  3.6,  0.7, -0.2],
    # p15: Relative mutability (Dayhoff)
    [100.,  20., 106., 102.,  41.,  49.,  66.,  96.,  56.,  40.,
      93.,  34.,  56.,  93.,  65., 120.,  97.,  74.,  18.,  41.],
    # p16: Alpha-helix propensity (Chou-Fasman)
    [1.42, 0.70, 1.01, 1.51, 1.13, 0.57, 1.00, 1.08, 1.16, 1.21,
     1.45, 0.67, 0.57, 1.11, 0.98, 0.77, 1.15, 1.06, 1.08, 0.69],
    # p17: Beta-sheet propensity (Chou-Fasman)
    [0.83, 1.19, 0.54, 0.37, 1.38, 0.75, 0.87, 1.60, 0.74, 1.30,
     1.05, 0.89, 0.55, 1.10, 0.93, 0.75, 0.70, 1.70, 1.37, 1.47],
    # p18: Turn propensity (Chou-Fasman)
    [0.66, 1.19, 1.46, 0.74, 0.60, 1.56, 0.95, 0.47, 1.01, 0.59,
     0.60, 1.56, 1.52, 0.98, 0.95, 1.43, 1.10, 0.50, 0.96, 0.98],
    # p19: Compressibility
    [7.56, 5.68, 4.61, 4.50, 3.71, 7.80, 5.03, 3.77, 3.42, 3.80,
     4.30, 5.55, 5.09, 4.71, 3.53, 6.73, 5.49, 4.13, 2.97, 4.00],
    # p20: Thermodynamic transfer hydrophobicity
    [ 0.44, 1.59, 0.59, 1.33, 2.04,-0.01, 0.88, 2.46, 1.04, 2.28,
      1.99, 0.90, 0.73, 1.28, 1.45, 0.44, 0.69, 1.79, 2.56, 1.26],
], dtype=np.float64)  # [20 properties, 20 amino acids]

# 氨基酸到索引的映射
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}


def compute_pseaac(sequence, omega=0.05, lam=20):
    """
    计算 PseAAC Type-2 特征 (420 维)

    Args:
        sequence: 氨基酸序列字符串
        omega: 序列顺序效应权重 (默认 0.05)
        lam: 滞后值 (默认 20)

    Returns:
        numpy array of shape (420,)
    """
    L = len(sequence)
    seq_indices = [AA_TO_IDX.get(aa, -1) for aa in sequence]

    # 1. 氨基酸组成 (20 dims)
    aac = np.zeros(20, dtype=np.float64)
    for idx in seq_indices:
        if idx >= 0:
            aac[idx] += 1
    aac = aac / L  # 归一化频率

    # 2. 序列顺序效应 (20 properties × 20 lags = 400 dims)
    theta = np.zeros((20, lam), dtype=np.float64)  # [20 props, 20 lags]

    for p in range(20):  # 每种性质
        prop_values = np.array([AA_PROPERTIES[p, idx] if idx >= 0 else 0.0
                                for idx in seq_indices])
        for lag in range(1, lam + 1):
            if L > lag:
                diff = prop_values[:L - lag] - prop_values[lag:]
                theta[p, lag - 1] = np.mean(diff ** 2)

    # 3. 组合 (PseAAC 标准公式)
    theta_sum = theta.sum()  # 所有 θ 的总和
    denom = 1.0 + omega * theta_sum

    # 前 20 个: 归一化 AAC
    pseaac = np.zeros(420, dtype=np.float64)
    pseaac[:20] = aac / denom

    # 后 400 个: 归一化 pseudo components
    theta_flat = theta.flatten()  # 400 dims
    pseaac[20:] = omega * theta_flat / denom

    return pseaac


# ===================== ProtBERT Token 特征提取 =====================

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def prepare_sequence(seq, max_length=51):
    """将序列转换为 ProtBERT 输入格式"""
    seq_spaced = " ".join(list(seq))
    seq_cleaned = re.sub(r"[UZOB]", "X", seq_spaced)
    return seq_cleaned


def extract_protbert_tokens(model, tokenizer, sequences, device, max_length=51, batch_size=16):
    """
    用微调后的 ProtBERT 提取 token-level 特征

    自动检测多GPU并使用 DataParallel 加速推理。

    Returns:
        token_embeddings: list of tensors, 每个 [L_i, 1024] (L_i 是实际序列长度)
        attention_masks: list of tensors
    """
    all_embeddings = []
    all_masks = []

    # 多GPU加速: 使用 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"  检测到 {torch.cuda.device_count()} 个GPU，启用 DataParallel")
        model = torch.nn.DataParallel(model)
        # DataParallel 后使用单GPU batch_size，总有效 batch_size = batch_size * n_gpus
        effective_batch_size = batch_size * torch.cuda.device_count()
        print(f"  有效 batch size: {effective_batch_size}")
    else:
        effective_batch_size = batch_size

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), effective_batch_size), desc="Extracting ProtBERT tokens"):
            batch_seqs = sequences[i:i + effective_batch_size]
            batch_processed = [prepare_sequence(s) for s in batch_seqs]

            encoded = tokenizer(
                batch_processed,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True
            )

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # DataParallel 会自动将 batch 分配到多个 GPU 并收集结果
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # DataParallel 返回的结果已经 gather 到主 GPU
            # outputs.last_hidden_state: [B, L_padded, 1024]
            hidden = outputs.last_hidden_state  # token-level!

            for j in range(len(batch_seqs)):
                mask_j = attention_mask[j].cpu()
                hidden_j = hidden[j].cpu()  # [L_padded, 1024]
                all_embeddings.append(hidden_j)
                all_masks.append(mask_j)

    return all_embeddings, all_masks


def save_token_embeddings(embeddings, masks, labels, sequences, output_path):
    """
    保存 token 嵌入到 HDF5 文件（增量写入，避免 OOM）

    零填充到最大长度，保存 mask 用于还原
    """
    import h5py
    import gc

    max_len = max(e.shape[0] for e in embeddings)
    dim = embeddings[0].shape[1]  # 1024
    n = len(embeddings)

    # 增量写入 HDF5，不预创建完整数组
    with h5py.File(output_path, 'w') as f:
        emb_dset = f.create_dataset(
            'embeddings', (n, max_len, dim), dtype='f4',
            chunks=(1, max_len, dim), compression='gzip', compression_opts=2
        )
        mask_dset = f.create_dataset(
            'masks', (n, max_len), dtype='i4',
            chunks=(1, max_len), compression='gzip', compression_opts=2
        )

        for i, (emb, mask) in enumerate(zip(embeddings, masks)):
            L = emb.shape[0]
            emb_dset[i, :L, :] = emb.numpy()
            mask_dset[i, :L] = mask.numpy()

        f.create_dataset('labels', data=np.array(labels, dtype=np.int32))
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('sequences', data=np.array(sequences, dtype=object), dtype=dt)

    # 释放内存
    del embeddings, masks
    gc.collect()

    print(f"  Token embeddings saved to {output_path}")
    print(f"    Shape: ({n}, {max_len}, {dim})")


def save_pseaac_features(pseaac_list, labels, sequences, output_path):
    """保存 PseAAC 特征到 CSV"""
    df = pd.DataFrame(pseaac_list)
    col = [f"PseAAC_{i + 1}" for i in range(df.shape[1])]
    df.columns = col
    df['label'] = labels

    # 只保存特征和标签（与原始格式一致：无 header, 标签在最后一列）
    df.to_csv(output_path, index=False, header=True)
    print(f"  PseAAC features saved to {output_path}")
    print(f"    Shape: {df.shape}")


# ===================== 主函数 =====================

def main():
    parser = argparse.ArgumentParser(description="提取 Token-level ProtBERT 特征 + 计算 PseAAC")
    parser.add_argument("--data_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--protbert_path", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "models" / "protbert_finetuned"))
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "features"))
    parser.add_argument("--max_length", type=int, default=51)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")

    # 加载微调后的 ProtBERT
    print(f"\n[1] 加载微调后的 ProtBERT: {args.protbert_path}")
    tokenizer = BertTokenizer.from_pretrained(args.protbert_path, do_lower_case=False)
    model = BertModel.from_pretrained(args.protbert_path)
    model.to(device)
    model.eval()
    print(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 处理 train/val/test
    for split in ['train', 'val', 'test']:
        csv_path = Path(args.data_dir) / f"{split}_dedup_split.csv"
        if not csv_path.exists():
            print(f"\n[跳过] {csv_path} 不存在")
            continue

        print(f"\n{'='*60}")
        print(f"处理 {split} 集: {csv_path}")
        print(f"{'='*60}")

        # 读取数据
        df = pd.read_csv(csv_path)
        sequences = df['Sequence'].tolist()
        labels = df['Label'].tolist()
        print(f"  序列数: {len(sequences)}")
        print(f"  最长序列: {max(len(s) for s in sequences)}")
        print(f"  正样本: {sum(labels)}, 负样本: {len(labels) - sum(labels)}")

        # 提取 ProtBERT token 特征
        print(f"\n  [1a] 提取 ProtBERT token-level 特征...")
        t0 = time.time()
        embeddings, masks = extract_protbert_tokens(
            model, tokenizer, sequences, device,
            max_length=args.max_length, batch_size=args.batch_size
        )
        t1 = time.time()
        print(f"  耗时: {t1 - t0:.1f}s")

        # 保存 token 嵌入
        token_path = output_dir / f"{split}_tokens.h5"
        save_token_embeddings(embeddings, masks, labels, sequences, token_path)

        # 计算 PseAAC
        print(f"\n  [1b] 计算 PseAAC 特征...")
        t0 = time.time()
        pseaac_list = []
        for seq in tqdm(sequences, desc=f"PseAAC ({split})"):
            pseaac_list.append(compute_pseaac(seq))
        t1 = time.time()
        print(f"  耗时: {t1 - t0:.1f}s")

        # 保存 PseAAC
        pseaac_path = output_dir / f"{split}_pseaac.csv"
        save_pseaac_features(pseaac_list, labels, sequences, pseaac_path)

        # 释放内存，防止 OOM
        del embeddings, masks, pseaac_list, sequences, labels
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("特征提取完成！")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
