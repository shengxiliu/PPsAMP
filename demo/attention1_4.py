'''
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Cross-Attention模型定义（保持不变）
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


# 加载测试特征的函数
def load_features(pseaac_file, protbert_file):
    df = pd.read_csv(pseaac_file, header=None)
    pseaac_features = torch.tensor(df.values, dtype=torch.float32)
    df1 = pd.read_hdf(protbert_file)
    protbert_features = torch.tensor(df1.values, dtype=torch.float32)
    return pseaac_features, protbert_features


# 加载测试数据
pseaac_file_test = "/home/zhaozhimiao/XZ/HLAB/4/feature/smorf_pseaac_features.csv"
protbert_file_test = "/home/zhaozhimiao/XZ/HLAB/4/feature/smorf.h5"
test_pseaac, test_protbert = load_features(pseaac_file_test, protbert_file_test)

# 假设有一个包含序列的文件，每行与特征对应
sequence_file = "/home/zhaozhimiao/XZ/HLAB/4/data/smorf_protein.csv"
# 读取序列文件
sequences = pd.read_csv(sequence_file, skiprows=1, header=None)[0].tolist()  # 提取序列列表
# 确保序列数量与测试集特征数量一致
print(f"序列文件行数: {len(sequences)}")
print(f"测试特征文件行数: {len(test_pseaac)}")

assert len(sequences) == len(test_pseaac), "序列文件的行数必须与测试集特征文件一致！"

batch_size = 32
test_dataset = TensorDataset(test_pseaac, test_protbert)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 模型初始化
embedding_dim = 256
num_heads = 4
num_layers = 2
pseaac_dim = 420
protbert_dim = 1024
num_classes = 2
dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CrossAttentionModel(embedding_dim, num_heads, num_layers, pseaac_dim, protbert_dim, num_classes, dropout)
model.load_state_dict(torch.load("/home/zhaozhimiao/XZ/HLAB/features/newnewnew_data/p3e110(2)/new/best_model.pth"))  # 加载已保存的模型
model.to(device)

# 测试模型并分类
model.eval()
positive_samples = []  # 存储正样本（类别 1）
negative_samples = []  # 存储负样本（类别 0）

with torch.no_grad():
    for batch_idx, (pseaac_batch, protbert_batch) in enumerate(tqdm(test_loader, desc="Testing")):
        pseaac_batch, protbert_batch = pseaac_batch.to(device), protbert_batch.to(device)
        outputs = model(pseaac_batch, protbert_batch)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

        # 对应的原始序列索引
        start_idx = batch_idx * batch_size
        end_idx = start_idx + len(predictions)
        batch_sequences = sequences[start_idx:end_idx]  # 从序列列表中提取

        # 按类别分开存储
        for i in range(len(predictions)):
            sample_info = {
                "Sequence": batch_sequences[i],          # 序列信息
                "Class_0_Prob": probabilities[i][0],    # 类别 0 概率
                "Class_1_Prob": probabilities[i][1],    # 类别 1 概率
            }
            if predictions[i] == 1:  # 类别 1（正样本）
                positive_samples.append(sample_info)
            else:  # 类别 0（负样本）
                negative_samples.append(sample_info)

# 保存测试集分类结果到 CSV 文件
positive_df = pd.DataFrame(positive_samples)
negative_df = pd.DataFrame(negative_samples)

positive_df.to_csv("/home/zhaozhimiao/XZ/HLAB/4/data/pre_positive.csv", index=False)
negative_df.to_csv("/home/zhaozhimiao/XZ/HLAB/4/data/pre_negative.csv", index=False)

print("分类结果已保存：")
print(f"正样本文件: /home/zhaozhimiao/XZ/HLAB/4/data/pre_positive.csv")
print(f"负样本文件: /home/zhaozhimiao/XZ/HLAB/4/data/pre_negative.csv")
'''

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import os
import h5py

# ==================== 数据集类 ====================
class FeatureDataset(Dataset):
    def __init__(self, pseaac_file, protbert_file):
        self.pseaac_df = pd.read_csv(pseaac_file)
        self.ids = self.pseaac_df.iloc[:, 0].values
        self.pseaac_features = self.pseaac_df.iloc[:, 1:].values

        with h5py.File(protbert_file, "r") as f:
            self.protbert_features = f["features"][:]

        # label 假设是 pseaac 文件最后一列
        self.labels = self.pseaac_df.iloc[:, -1].values

        # 检查 id 长度匹配
        assert len(self.ids) == len(self.protbert_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.ids[idx],
            torch.tensor(self.pseaac_features[idx], dtype=torch.float32),
            torch.tensor(self.protbert_features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ==================== 模型 ====================
class AttentionModel(nn.Module):
    def __init__(self, pseaac_dim, protbert_dim=1024, hidden_dim=256, num_classes=2):
        super(AttentionModel, self).__init__()
        self.pseaac_projection = nn.Linear(pseaac_dim, hidden_dim)
        self.protbert_projection = nn.Linear(protbert_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, pseaac_features, protbert_features):
        pseaac_proj = self.pseaac_projection(pseaac_features).unsqueeze(1)
        protbert_proj = self.protbert_projection(protbert_features).unsqueeze(1)
        combined = torch.cat([pseaac_proj, protbert_proj], dim=1)
        attn_output, _ = self.attention(combined, combined, combined)
        attn_output = attn_output.mean(dim=1)
        return self.fc(attn_output)

# ==================== 工具函数 ====================
def get_predictions(loader, model, device):
    model.eval()
    all_ids, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for ids, pseaac, protbert, labels in loader:
            pseaac, protbert, labels = pseaac.to(device), protbert.to(device), labels.to(device)
            outputs = model(pseaac, protbert)
            preds = torch.argmax(outputs, dim=1)
            all_ids.extend(ids)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    return all_ids, all_labels, all_preds

# ==================== 主程序 ====================
if __name__ == "__main__":
    # ------------------- 数据路径 ------------------- #
    lsx_root = "/media/ubuntu/conda/lsx/AMP"

    # Yan 数据集
    pseaac_file_train_yan = os.path.join(lsx_root, "PseAAC/Yan/train.csv")
    protbert_file_train_yan = os.path.join(lsx_root, "p3e110(2)/Yan/train.h5")
    pseaac_file_val_yan = os.path.join(lsx_root, "PseAAC/Yan/val.csv")
    protbert_file_val_yan = os.path.join(lsx_root, "p3e110(2)/Yan/val.h5")
    pseaac_file_test_yan = os.path.join(lsx_root, "PseAAC/Yan/test.csv")
    protbert_file_test_yan = os.path.join(lsx_root, "p3e110(2)/Yan/test.h5")

    # New 数据集
    pseaac_file_train_new = os.path.join(lsx_root, "PseAAC/new/train.csv")
    protbert_file_train_new = os.path.join(lsx_root, "p3e110(2)/new/train.h5")
    pseaac_file_val_new = os.path.join(lsx_root, "PseAAC/new/val.csv")
    protbert_file_val_new = os.path.join(lsx_root, "p3e110(2)/new/val.h5")
    pseaac_file_test_new = os.path.join(lsx_root, "PseAAC/new/test.csv")
    protbert_file_test_new = os.path.join(lsx_root, "p3e110(2)/new/test.h5")

    output_dir = lsx_root

    # 选择数据集 (Yan / New)
    use_yan = True  # 改成 False 就切换到 New

    if use_yan:
        pseaac_train, protbert_train = pseaac_file_train_yan, protbert_file_train_yan
        pseaac_val, protbert_val = pseaac_file_val_yan, protbert_file_val_yan
        pseaac_test, protbert_test = pseaac_file_test_yan, protbert_file_test_yan
        dataset_name = "Yan"
    else:
        pseaac_train, protbert_train = pseaac_file_train_new, protbert_file_train_new
        pseaac_val, protbert_val = pseaac_file_val_new, protbert_file_val_new
        pseaac_test, protbert_test = pseaac_file_test_new, protbert_file_test_new
        dataset_name = "New"

    # 自动检测 PseAAC 特征维度
    pseaac_dim = pd.read_csv(pseaac_train).shape[1] - 2  # 去掉 id 和 label
    print(f"Detected PseAAC feature dimension: {pseaac_dim}")

    # 数据加载
    train_dataset = FeatureDataset(pseaac_train, protbert_train)
    val_dataset = FeatureDataset(pseaac_val, protbert_val)
    test_dataset = FeatureDataset(pseaac_test, protbert_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionModel(pseaac_dim=pseaac_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for ids, pseaac, protbert, labels in train_loader:
            pseaac, protbert, labels = pseaac.to(device), protbert.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pseaac, protbert)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # 测试集预测
    ids, true_labels, preds = get_predictions(test_loader, model, device)

    # 保存预测结果
    results_df = pd.DataFrame({
        "id": ids,
        "true_label": true_labels,
        "predicted_label": preds
    })
    results_file = os.path.join(output_dir, f"test_predictions_{dataset_name}.csv")
    results_df.to_csv(results_file, index=False)
    print(f"✅ Predictions saved to {results_file}")

    # 评估
    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds)
    rec = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    mcc = matthews_corrcoef(true_labels, preds)
    cm = confusion_matrix(true_labels, preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MCC: {mcc:.4f}")
    print("Confusion Matrix:")
    print(cm)
