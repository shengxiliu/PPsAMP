#!/bin/bash

# =================================================================
# PPsAMP 一键预测流水线 
# 用法: bash predict.sh <输入fasta> <输出csv>
# =================================================================

# 1. 环境变量配置 (解决国内下载卡顿和环境警告问题)
export HF_ENDPOINT=https://hf-mirror.com
export NUMEXPR_MAX_THREADS=16

# 2. 参数校验
if [ "$#" -lt 2 ]; then
    echo "❌ 错误：参数不足！"
    echo "👉 正确用法: bash predict.sh <输入文件.fasta> <输出结果.csv>"
    exit 1
fi

INPUT_FASTA=$(realpath "$1")
OUTPUT_CSV=$(realpath "$2")
BASE_DIR=$(pwd)
TEMP_DIR="$BASE_DIR/temp_run"

# 设置模型默认路径
BERT_DIR="$BASE_DIR/finetuned_protbert"
MODEL_PTH="$BASE_DIR/best_model_e20p5.pth"

# 3. 文件完整性检查 (保姆级防呆)
if [ ! -f "$MODEL_PTH" ]; then
    echo "❌ 错误: 找不到第三步的预测模型权重！"
    echo "请去 GitHub Releases 下载 best_model_e20p5.pth 并将其放在当前主目录下: $BASE_DIR"
    exit 1
fi

if [ ! -d "$BERT_DIR" ] || [ ! -f "$BERT_DIR/config.json" ]; then
    echo "❌ 错误: 找不到第二步的 ProtBERT 微调模型文件夹！"
    echo "请在当前目录下新建一个名为 'finetuned_protbert' 的文件夹，"
    echo "并将 Releases 中的 model.safetensors, config.json, vocab.txt 等 6 个文件放入其中。"
    exit 1
fi

# 4. 创建临时文件夹
mkdir -p "$TEMP_DIR"

# 5. 执行流水线
echo "✅ 模型检查通过！开始执行流水线..."

echo "[1/3] Extracting PseAAC features..."
python feature-pseaac/PseAAC_1_4.py --fasta "$INPUT_FASTA" --output "$TEMP_DIR/pseaac.csv"
if [ $? -ne 0 ]; then echo "❌ PseAAC 特征提取失败，请检查输入序列格式！"; exit 1; fi

echo "[2/3] Extracting ProtBERT features using local finetuned model..."
# 这里自动指向了本地你微调好的文件夹
python feature-t/feature_extract_test2.py --fasta "$INPUT_FASTA" --model_name "$BERT_DIR" --output "$TEMP_DIR/bert.h5"
if [ $? -ne 0 ]; then echo "❌ ProtBERT 特征提取失败！"; exit 1; fi

echo "[3/3] Running Inference..."
python demo/attention1_4.py \
    --pseaac_input "$TEMP_DIR/pseaac.csv" \
    --bert_input "$TEMP_DIR/bert.h5" \
    --model_path "$MODEL_PTH" \
    --output "$OUTPUT_CSV"

# 6. 清理与完成
if [ -f "$OUTPUT_CSV" ]; then
    echo "🎉 预测大功告成！结果已保存至: $OUTPUT_CSV"
    # 清理中间产生的临时文件
    rm -rf "$TEMP_DIR"
else
    echo "⚠️ 预测可能失败，未找到输出的 CSV 文件。"
fi
