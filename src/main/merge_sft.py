from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# 加载 SFT 训练出的带有 LoRA 的模型
model = AutoPeftModelForCausalLM.from_pretrained(
    "output/SFT/config_data_SFT_final", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("output/SFT/config_data_SFT_final")

# 将 LoRA 权重合并到主干网络中
model = model.merge_and_unload()

# 保存合并后的完整模型
model.save_pretrained("output/SFT/merged_model")
tokenizer.save_pretrained("output/SFT/merged_model")
