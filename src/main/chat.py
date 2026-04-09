import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
LORA_PATH = "output/terraria_lora_models/Andrew_V1"

# 1. 加载基础模型和 Tokenizer
print("loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", dtype=torch.bfloat16
)

# 2. 挂载 LoRA 权重
model = PeftModel.from_pretrained(base_model, LORA_PATH)
print("model loaded and LoRA weights applied successfully!\n")
print("-" * 50)
print(
    "Welcome to the Terraria Chat Assistant! Ask me anything about the game, and I'll do my best to help you out."
)

# 3. 初始化带有 System Prompt 的对话历史
messages = [{"role": "system", "content": "You are a Terraria game assistant."}]

# 4. 开启无限循环，实现连续对话
while True:
    user_input = input("\nUser : ")

    if user_input.strip().lower() in ["quit", "exit"]:
        print("done")
        break

    messages.append({"role": "user", "content": user_input})

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    print(f"\nAndrew: {response}")

    messages.append({"role": "assistant", "content": response})
