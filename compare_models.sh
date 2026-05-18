#!/bin/bash
# 比较SFT和DPO模型的评估脚本

echo "开始比较不同微调方法的模型..."

# 评估LoRA SFT模型
echo "评估 LoRA SFT 模型..."
python run_advanced_eval.py \
  --base_model_path meta-llama/Llama-3.2-3B-Instruct \
  --sft_model_path output/terraria_lora_models/Andrew_V1_final \
  --eval_data_path eval_data.json \
  --output_path eval_results_sft.json

# 评估DPO模型
echo "评估 DPO 模型..."
python run_advanced_eval.py \
  --base_model_path meta-llama/Llama-3.2-3B-Instruct \
  --sft_model_path output/terraria_dpo_model \
  --eval_data_path eval_data.json \
  --output_path eval_results_dpo.json

echo "比较评估完成！"
echo "SFT结果保存在 eval_results_sft.json"
echo "DPO结果保存在 eval_results_dpo.json"