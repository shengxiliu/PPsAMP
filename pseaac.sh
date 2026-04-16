#!/bin/bash

# =================================================================
# PPsAMP 一键预测流水线 (GitHub 版)
# 用法: ./predict.sh <输入fasta> <输出csv> [模型权重路径]
# =================================================================

# 1. 参数校验与路径设置
if [ "$#" -lt 2 ]; then
    echo "Usage: ./predict.sh <input.fasta> <output_results.csv> [model_path]"
    echo "Default model_path: ./best_model_e20p5.pth"
    exit 1
fi

INPUT_FASTA=$(realpath "$1")
OUTPUT_CSV=$(realpath "$2")
# 如果用户提供了第三个参数，则使用它；否则使用默认的 ./best_model_e20p5.pth
MODEL_PTH=$(realpath "${3:-./best_model_e20p5.pth}")

BASE_DIR=$(pwd)
TEMP_DIR="$BASE_DIR/temp_run"
mkdir -p "$TEMP_DIR"

# 检查模型是否存在
if [ ! -f "$MODEL_PTH" ]; then
    echo "Error: Model file not found at $MODEL_PTH"
    echo "Please download the weights from Releases and place them in the root directory,"
    echo "or specify the path as the 3rd argument."
    exit 1
fi

# 2. 执行流程
echo "Using model: $MODEL_PTH"
echo "[1/3] Extracting PseAAC features..."
python feature-pseaac/PseAAC_1_4.py --fasta "$INPUT_FASTA" --output "$TEMP_DIR/pseaac.csv"

echo "[2/3] Extracting ProtBERT features..."
python feature-t/feature_extract_test2.py --fasta "$INPUT_FASTA" --model_name prot_bert_bfd --output "$TEMP_DIR/bert.h5"

echo "[3/3] Running Inference..."
python demo/attention1_4.py \
    --pseaac_input "$TEMP_DIR/pseaac.csv" \
    --bert_input "$TEMP_DIR/bert.h5" \
    --model_path "$MODEL_PTH" \
    --output "$OUTPUT_CSV"

# 3. 清理与预览
if [ -f "$OUTPUT_CSV" ]; then
    echo "Done! Results saved to $OUTPUT_CSV"
    rm -rf "$TEMP_DIR"
fi