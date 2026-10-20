#!/usr/bin/env python3
"""
第二步：训练 Token-Level Cross-Attention 模型

架构:
  Q = PseAAC (420-dim → project → 256-dim, 1 token)
  K = ProtBERT per-token (1024-dim → project → 256-dim, L tokens)
  V = ProtBERT per-token (1024-dim → project → 256-dim, L tokens)

  Cross-Attention → Self-Attention × num_layers → FFN → Classifier

输入: features/train_tokens.h5 + features/train_pseaac.csv (及 val)
输出: models/best_model.pt
"""

import argparse, os, time
import numpy as np
import h5py
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             roc_auc_score)
from pathlib import Path
from tqdm import tqdm


# ===================== 模型定义 =====================

class TokenCrossAttentionModel(nn.Module):
    """
    Q = PseAAC (1 个全局 token)
    K = ProtBERT token-level 特征 (L 个氨基酸 token)
    V = ProtBERT token-level 特征 (L 个氨基酸 token)

    → Cross-attention 得到 [B, 1, L] 的注意力权重
    → 每个氨基酸位置获得一个注意力分数
    """
    def __init__(self, embedding_dim=256, num_heads=4, num_layers=2,
                 pseaac_dim=420, protbert_dim=1024, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 特征投影
        self.pseaac_projection = nn.Linear(pseaac_dim, embedding_dim)
        self.protbert_projection = nn.Linear(protbert_dim, embedding_dim)

        # Cross-Attention: Q = PseAAC, K/V = ProtBERT tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # 使用 batch_first 简化维度操作
        )

        # Self-Attention 层（对 cross-attention 输出做进一步处理）
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        # Feed-Forward 层
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, 4 * embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * embedding_dim, embedding_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # Layer Norm
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_layers)
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, pseaac_features, protbert_tokens, protbert_mask=None,
                return_attn_weights=False):
        """
        Args:
            pseaac_features: [B, 420] PseAAC 特征
            protbert_tokens: [B, L, 1024] ProtBERT per-token 特征
            protbert_mask:  [B, L] 有效 token mask (1=有效, 0=padding)
            return_attn_weights: 是否返回 cross-attention 权重

        Returns:
            logits: [B, num_classes]
            cross_attn_weights: [B, num_heads, 1, L] (如果 return_attn_weights=True)
        """
        B, L, _ = protbert_tokens.shape

        # 投影到共同维度
        query = self.pseaac_projection(pseaac_features).unsqueeze(1)  # [B, 1, 256]
        kv = self.protbert_projection(protbert_tokens)                 # [B, L, 256]

        # 构建 key_padding_mask (True = ignore)
        key_padding_mask = None
        if protbert_mask is not None:
            key_padding_mask = (protbert_mask == 0)  # [B, L]

        # Cross-Attention
        attn_output, cross_attn_weights = self.cross_attention(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # 保留每个 head 的权重 [B, num_heads, 1, L]
        )
        # attn_output: [B, 1, 256], cross_attn_weights: [B, num_heads, 1, L]

        # Self-Attention + FFN 层
        for i in range(self.num_layers):
            # Self-attention (attn_output 本身做 self-attention, seq_len=1)
            sa_output, _ = self.self_attention_layers[i](
                query=attn_output,
                key=attn_output,
                value=attn_output,
                need_weights=False
            )
            attn_output = self.attn_norms[i](attn_output + sa_output)

            # Feed-forward
            ff_output = self.feed_forward_layers[i](attn_output)
            attn_output = self.ffn_norms[i](attn_output + ff_output)

        # 分类
        pooled = attn_output.squeeze(1)  # [B, 256]
        logits = self.classifier(pooled)  # [B, 2]

        if return_attn_weights:
            return logits, cross_attn_weights
        return logits


# ===================== 数据集 =====================

class TokenFeatureDataset(Dataset):
    def __init__(self, tokens_h5_path, pseaac_csv_path):
        # 加载 ProtBERT token 特征
        with h5py.File(tokens_h5_path, 'r') as f:
            self.protbert_tokens = torch.tensor(f['embeddings'][:], dtype=torch.float32)
            self.masks = torch.tensor(f['masks'][:], dtype=torch.int32)
            self.labels = torch.tensor(f['labels'][:], dtype=torch.long)

        # 加载 PseAAC 特征（CSV 中是 float64，必须强转 float32）
        df = pd.read_csv(pseaac_csv_path)
        pseaac_cols = [c for c in df.columns if c.startswith('PseAAC_')]
        self.pseaac = torch.from_numpy(df[pseaac_cols].values.astype(np.float32))

        assert len(self.pseaac) == len(self.protbert_tokens), \
            f"PseAAC ({len(self.pseaac)}) vs ProtBERT ({len(self.protbert_tokens)}) 样本数不一致!"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.pseaac[idx].float(), self.protbert_tokens[idx].float(),
                self.masks[idx], self.labels[idx])


# ===================== 评估 =====================

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for pseaac, tokens, masks, labels in loader:
            pseaac = pseaac.float().to(device)
            tokens = tokens.float().to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = model(pseaac, tokens, masks)
            loss = criterion(logits.float(), labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return {
        'loss': total_loss / len(loader),
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'specificity': spec, 'f1': f1, 'mcc': mcc, 'auc': auc,
        'predictions': all_preds, 'probs': all_probs, 'labels': all_labels,
        'confusion': (tn, fp, fn, tp)
    }


# ===================== 训练 =====================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for pseaac, tokens, masks, labels in tqdm(loader, desc="Training", leave=False):
        pseaac = pseaac.to(device)
        tokens = tokens.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(pseaac.float(), tokens.float(), masks)
        loss = criterion(logits.float(), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser(description="训练 Token-Level Cross-Attention 模型")
    parser.add_argument("--feature_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "features"))
    parser.add_argument("--output_dir", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "models" / "token_attention"))
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"设备: {device}")
    print(f"特征目录: {args.feature_dir}")
    print(f"输出目录: {args.output_dir}")

    # 加载数据
    feature_dir = Path(args.feature_dir)
    train_dataset = TokenFeatureDataset(
        feature_dir / "train_tokens.h5",
        feature_dir / "train_pseaac.csv"
    )
    val_dataset = TokenFeatureDataset(
        feature_dir / "val_tokens.h5",
        feature_dir / "val_pseaac.csv"
    )
    test_dataset = TokenFeatureDataset(
        feature_dir / "test_tokens.h5",
        feature_dir / "test_pseaac.csv"
    )

    print(f"\n数据集大小:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")
    print(f"  ProtBERT token shape: {train_dataset.protbert_tokens.shape}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size * 2)

    # 创建模型
    model = TokenCrossAttentionModel(
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        pseaac_dim=420,
        protbert_dim=1024,
        num_classes=2,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数: {total_params:,} (trainable: {trainable_params:,})")

    # 类别权重 (处理不平衡)
    train_labels = train_dataset.labels.numpy()
    n_pos = (train_labels == 1).sum()
    n_neg = (train_labels == 0).sum()
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"  类别分布: 负={n_neg}, 正={n_pos}, pos_weight={pos_weight:.2f}")

    class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # 训练
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"开始训练 ({args.epochs} epochs, patience={args.patience})")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_results = evaluate(model, train_loader, device, criterion)
        val_results = evaluate(model, val_loader, device, criterion)
        t1 = time.time()

        scheduler.step(val_results['f1'])

        print(f"Epoch {epoch:3d}/{args.epochs} ({t1-t0:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val F1: {val_results['f1']:.4f} | "
              f"Val AUC: {val_results['auc']:.4f} | "
              f"Val Acc: {val_results['accuracy']:.4f}")

        if val_results['f1'] > best_val_f1:
            best_val_f1 = val_results['f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  ✓ 保存最佳模型 (Val F1={best_val_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 在测试集上评估最佳模型
    print(f"\n{'='*60}")
    print(f"加载最佳模型 (epoch {best_epoch}, Val F1={best_val_f1:.4f})")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_results = evaluate(model, test_loader, device, criterion)

    print(f"\nTest Results:")
    print(f"  Accuracy:    {test_results['accuracy']:.4f}")
    print(f"  Precision:   {test_results['precision']:.4f}")
    print(f"  Recall:      {test_results['recall']:.4f}")
    print(f"  Specificity: {test_results['specificity']:.4f}")
    print(f"  F1 Score:    {test_results['f1']:.4f}")
    print(f"  MCC:         {test_results['mcc']:.4f}")
    print(f"  AUC:         {test_results['auc']:.4f}")
    print(f"  Confusion: TN={test_results['confusion'][0]}, "
          f"FP={test_results['confusion'][1]}, "
          f"FN={test_results['confusion'][2]}, "
          f"TP={test_results['confusion'][3]}")

    # 保存结果
    results_df = pd.DataFrame([{
        'best_epoch': best_epoch,
        'val_f1': best_val_f1,
        'test_accuracy': test_results['accuracy'],
        'test_precision': test_results['precision'],
        'test_recall': test_results['recall'],
        'test_specificity': test_results['specificity'],
        'test_f1': test_results['f1'],
        'test_mcc': test_results['mcc'],
        'test_auc': test_results['auc'],
    }])
    results_df.to_csv(output_dir / "training_results.csv", index=False)
    print(f"\n结果已保存到: {output_dir / 'training_results.csv'}")


if __name__ == "__main__":
    main()
