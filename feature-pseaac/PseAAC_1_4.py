import numpy as np
import csv
from tqdm import tqdm  # 导入 tqdm 模块

# 定义氨基酸的属性字典（Hydrophobicity，极性，电荷，等等）
amino_acid_properties = {
    'A': [1.8, 91.5, 0, 0, 15, -0.5, 27.8, 4.34, 705.42, 31.5, 70.079, 0.48], 'C': [2.5, 118, 1.48, 8.18, 5, -1, 15.5, 35.77, 2412.56, 13.9, 103.144, 0.32], 
    'D': [-3.5, 124.5, 40.7, 3.65, 50, 3, 60.6, 12, 34.96, 60.9, 115.089, 0.81], 'E': [-3.5, 115.1, 49.91, 4.25, 55, 3, 68.2, 17.26, 1158.66, 72.3, 129.116, 0.93],
    'F': [2.8, 203.4, 0.35, 0, 10, -2.5, 25.5, 29.4, 5203.86, 28.7, 147.177, 0.42], 'G': [-0.4, 66.4, 0, 0, 10, 0, 24.5, 0, 33.18, 25.2, 57.052, 0.51],
    'H': [-3.2, 167.3, 51.6, 6, 34, -0.5, 50.7, 21.81, 1637.13, 46.7, 137.142, 0.66], 'I': [4.5, 168.8, 0.15, 0, 13, -1.8, 22.8, 19.06, 5979.4, 23, 113.16, 0.39],
    'K': [-3.9, 171.3, 49.5, 10.53, 85, 3, 103, 21.29, 699.69, 110.3, 128.174, 0.93], 'L': [3.8, 167.9, 0.45, 0, 16, -1.8, 27.6, 18.78, 4985.7, 29, 113.16, 0.41],
    'M': [1.9, 170.8, 1.43, 0, 20, -1.3, 33.5, 21.64, 4491.66, 30.5, 131.198, 0.44], 'N': [-3.5, 135.2, 3.38, 0, 49, 0.2, 60.1, 13.28, 513.46, 62.2, 114.104, 0.82],
    'P': [-1.6, 129.3, 1.58, 0, 45, 0, 51.5, 10.93, 431.96, 53.7, 97.177, 0.78], 'Q': [-3.5, 161.1, 3.53, 0, 56, 0.2, 68.7, 17.56, 1087.83, 74, 128.131, 0.81], 
    'R': [-4.5, 202, 52, 12.48, 67, 3, 94.7, 26.66, 1484.28, 93.8, 156.188, 0.84], 'S': [-0.8, 99.1, 1.67, 0, 32, 0.3, 42, 6.35, 174.76, 44.2, 87.078, 0.7], 
    'T': [-0.7, 122.1, 1.66, 0, 32, -0.4, 45, 11.01, 601.88, 46, 101.105, 0.71], 'V': [4.2, 141.7, 0.13, 0, 14, -1.5, 23.7, 13.92, 4474.4, 23.5, 99.133, 0.4],
    'W': [-0.9, 237.6, 2.1, 0, 17, -3.4, 34.7, 42.53, 6374.07, 41.7, 186.213, 0.49], 'Y': [-1.3, 203.6, 1.61, 10.7, 41, -2.3, 55.2, 31.53, 4291.1, 59.1, 163.17, 0.67],
    'X': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

# 读取.csv文件并提取序列
def read_sequences_from_csv(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sequence = row["Sequence"]
            # 将长度小于30的序列用 'X' 填充至30长度
            sequence_padded = sequence + 'X' * (30 - len(sequence))
            sequences.append(sequence_padded)
    return sequences

# 保存特征到.csv文件
def save_features_to_csv(features, file_path):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for feature in features:
            writer.writerow(list(map(str, feature)))

# 定义计算PseAAC特征的函数
def compute_pse_aac(sequence, k=2, lambda_=5):
    features = []

    # 遍历每个物理化学属性
    for prop_index in range(12):
        # 计算AAC部分
        prop_features = [
            amino_acid_properties.get(aa, [0] * 12)[prop_index] for aa in sequence
        ]

        # 计算序列相关性（theta 部分）
        theta = []
        for lam in range(1, lambda_ + 1):
            if len(sequence) <= lam:
                theta.append(0)
            else:
                theta_value = sum(
                    (amino_acid_properties.get(sequence[i], [0] * 12)[prop_index] -
                     amino_acid_properties.get(sequence[i + lam], [0] * 12)[prop_index]) ** 2
                    for i in range(len(sequence) - lam)
                )
                theta.append(theta_value / (len(sequence) - lam))

        # 合并 AAC 和 theta
        features.extend(prop_features + theta)

    # 对整体特征进行归一化
    features = np.array(features, dtype=np.float32)
    norm = np.linalg.norm(features)
    if norm != 0:
        features /= norm
    else:
        print("Warning: Zero norm detected in features.")

    return features

# 示例.csv文件路径
csv_file_path = '/home/zhaozhimiao/XZ/HLAB/4/data/smorf_protein1.csv'

# 读取序列
sequences = read_sequences_from_csv(csv_file_path)

# 计算每个序列的PseAAC特征并保存
all_features = []
for sequence in tqdm(sequences, desc="Computing PseAAC Features"):
    pse_aac_features = compute_pse_aac(sequence, k=2)
    all_features.append(pse_aac_features)

# 保存特征到.csv文件
output_file_path = '/home/zhaozhimiao/XZ/HLAB/4/feature/smorf_pseaac_nocdhit.csv'
save_features_to_csv(all_features, output_file_path)

print(f"PseAAC 特征已保存至 {output_file_path}")
