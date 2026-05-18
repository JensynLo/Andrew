import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

# ================= 配置区 =================
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR_BASE = "./output/terraria_lora_models"
MAX_SEQ_LENGTH = 2048
WIKI_DATA_RATIO = 1.0


# ================= 1. 数据清洗与对齐 =================
def format_to_llama3_template(example, tokenizer):
    """process custom wiki json data"""
    if "instruction" in example and "output" in example:
        messages = [
            {"role": "system", "content": example.get("system", "")},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
    elif "messages" in example:
        messages = [{"role": "system", "content": example.get("system", "")}] + example[
            "messages"
        ]
    else:
        messages = []

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def format_wiki_to_llama3(example, tokenizer):
    """
    process the lparkourer10/terraria-wiki dataset, which has a different structure from our custom data.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a Terraria Wiki assistant. Provide accurate and helpful answers based on the game's official wiki and history.",
        },
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def load_and_prepare_data(file_path, tokenizer):
    ds = load_dataset("json", data_files=file_path, split="train")
    ds = ds.map(lambda x: format_to_llama3_template(x, tokenizer), num_proc=4)
    ds = ds.filter(lambda x: len(tokenizer(x["text"])["input_ids"]) <= MAX_SEQ_LENGTH)
    return ds


def load_and_prepare_hf_wiki(tokenizer):
    print("downloading lparkourer10/terraria-wiki ...")
    ds = load_dataset("lparkourer10/terraria-wiki", split="train")
    ds = ds.map(lambda x: format_wiki_to_llama3(x, tokenizer), num_proc=8)
    ds = ds.filter(lambda x: len(tokenizer(x["text"])["input_ids"]) <= MAX_SEQ_LENGTH)
    print(f"Wiki with {len(ds)} records")
    return ds


# ================= 2. 核心微调函数 =================
def run_finetuning(phase_name, train_dataset, model_id, tokenizer):
    print(
        f"\n{'=' * 50}\n🚀 training: {phase_name} | with: {len(train_dataset)}\n{'=' * 50}"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()

    # ========================================

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=os.path.join(OUTPUT_DIR_BASE, phase_name),
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        num_train_epochs=3,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=config,
    )

    trainer.train()
    final_save_path = os.path.join(OUTPUT_DIR_BASE, f"{phase_name}_final")
    trainer.model.save_pretrained(final_save_path)  # type: ignore
    tokenizer.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}\n")

    del model
    del trainer
    torch.cuda.empty_cache()


# ================= 3. 主流程执行 =================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("loading custom data ...")
    ds_how_to = load_and_prepare_data("output/how_to_get_qa.jsonl", tokenizer)
    ds_blind = load_and_prepare_data("output/blind_qa.jsonl", tokenizer)
    ds_multiturn = load_and_prepare_data("output/multiturn_dialogues.jsonl", tokenizer)

    ds_custom = concatenate_datasets([ds_how_to, ds_blind, ds_multiturn])
    custom_size = len(ds_custom)
    print(f"Custom data: {custom_size} records")

    ds_wiki_full = load_and_prepare_hf_wiki(tokenizer)

    target_wiki_size = int(custom_size * WIKI_DATA_RATIO)
    target_wiki_size = min(target_wiki_size, len(ds_wiki_full))

    ds_wiki_sampled = ds_wiki_full.shuffle(seed=42).select(range(target_wiki_size))
    final_ds = concatenate_datasets([ds_custom, ds_wiki_sampled]).shuffle(seed=42)
    print(
        f"\n final: {len(ds_custom) + len(ds_wiki_sampled)} records (Custom: {len(ds_custom)}, Wiki: {len(ds_wiki_sampled)})"
    )
    run_finetuning("Andrew_V1", final_ds, MODEL_ID, tokenizer)


if __name__ == "__main__":
    main()
