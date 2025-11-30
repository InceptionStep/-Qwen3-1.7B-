#!/bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)/src


# 创建输出目录
mkdir -p output logs

echo "🚀 开始执行 Qwen3 Medical SFT 项目..."

# 1. 数据准备
echo "📋 正在准备数据..."
python src/data.py

# 2. 全量微调训练
echo "🏋️ 正在进行全量微调训练..."
python src/train.py > logs/train.log 2>&1

# 3. LoRA 微调训练
echo "🎯 正在进行 LoRA 微调训练..."
python src/train_lora.py > logs/train_lora.log 2>&1

# 4. 模型推理测试
echo "🔍 运行推理测试..."
python src/inference.py > logs/inference.log 2>&1
python src/inference_lora.py > logs/inference_lora.log 2>&1

# 5. 模型评估
echo "📊 运行模型评估..."
python src/wj_evaluate_lora.py > logs/evaluate_lora.log 2>&1
python src/wj_lora_roung_.py > logs/rouge_evaluate.log 2>&1

echo "✅ 所有任务已完成！"