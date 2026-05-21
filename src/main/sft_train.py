import os
import wandb
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from ..utils import load_config


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


def load_and_prepare_data(file_path, tokenizer, MAX_SEQ_LENGTH):
    ds = load_dataset("json", data_files=file_path, split="train")
    ds = ds.map(lambda x: format_to_llama3_template(x, tokenizer), num_proc=4)
    ds = ds.filter(lambda x: len(tokenizer(x["text"])["input_ids"]) <= MAX_SEQ_LENGTH)
    return ds


def load_and_prepare_hf_wiki(tokenizer, MAX_SEQ_LENGTH):
    print("downloading lparkourer10/terraria-wiki ...")
    ds = load_dataset("lparkourer10/terraria-wiki", split="train")
    ds = ds.map(lambda x: format_wiki_to_llama3(x, tokenizer), num_proc=8)
    ds = ds.filter(lambda x: len(tokenizer(x["text"])["input_ids"]) <= MAX_SEQ_LENGTH)
    print(f"Wiki with {len(ds)} records")
    return ds


# ================= 2. 核心微调函数 =================
def run_finetuning(phase_name, train_dataset, validation_dataset, tokenizer, configs):
    print(
        f"\n{'=' * 50}\n🚀 training: {phase_name} | with: {len(train_dataset)}\n{'=' * 50}"
    )

    # 1. 初始化 wandb
    wandb.init(
        project=configs["wandb_project"],
        name=phase_name,
        config=configs,
    )

    model = AutoModelForCausalLM.from_pretrained(
        configs["model"]["model_id"],
        device_map={"": 0},
        dtype=configs["model"]["torch_dtype"],
    )
    if configs["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # ========================================

    peft_config = LoraConfig(
        r=configs["training"]["peft_config"]["r"],
        lora_alpha=configs["training"]["peft_config"]["lora_alpha"],
        target_modules=configs["training"]["peft_config"]["target_modules"],
        lora_dropout=configs["training"]["peft_config"]["lora_dropout"],
        bias=configs["training"]["peft_config"]["bias"],
        task_type=configs["training"]["peft_config"]["task_type"],
    )

    sft_config_args = configs["training"]["sft_config"].copy()
    sft_config_args["output_dir"] = os.path.join(
        configs["training"]["output_dir_base"], phase_name
    )
    # 强制设置 report_to 参数
    sft_config_args["report_to"] = "wandb"

    config = SFTConfig(**sft_config_args)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=config,
    )

    trainer.train()
    final_save_path = os.path.join(
        configs["training"]["output_dir_base"], f"{phase_name}_final"
    )
    trainer.model.save_pretrained(final_save_path)  # type: ignore
    tokenizer.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}\n")

    del model
    del trainer
    torch.cuda.empty_cache()

    # 2. 结束当前的 wandb run，以便进入下一个 phase
    wandb.finish()


# ================= 3. 主流程执行 =================
def main():
    parser = argparse.ArgumentParser(description="SFT Training Script")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_train_configs.json",
        help="Path to config file",
    )
    args = parser.parse_args()
    configs = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(configs["model"]["model_id"])
    print("loading custom data ...")
    ds_list = []
    for data_file in configs["data"]["custom_data_files"]:
        ds = load_and_prepare_data(
            data_file, tokenizer, configs["training"]["max_seq_length"]
        )
        ds_list.append(ds)

    ds_custom = concatenate_datasets(ds_list)
    custom_size = len(ds_custom)
    print(f"Custom data: {custom_size} records")

    ds_wiki_full = load_and_prepare_hf_wiki(
        tokenizer, configs["training"]["max_seq_length"]
    )

    target_wiki_size = int(custom_size * configs["training"]["wiki_data_ratio"])
    target_wiki_size = min(target_wiki_size, len(ds_wiki_full))

    ds_wiki_sampled = ds_wiki_full.shuffle(seed=42).select(range(target_wiki_size))
    final_ds = concatenate_datasets([ds_custom, ds_wiki_sampled]).shuffle(seed=42)
    print(
        f"\n final: {len(ds_custom) + len(ds_wiki_sampled)} records (Custom: {len(ds_custom)}, Wiki: {len(ds_wiki_sampled)})"
    )
    # 分割训练和验证集
    train_size = int(0.95 * len(final_ds))
    final_ds = final_ds.shuffle(seed=42)
    split_ds = final_ds.train_test_split(train_size=train_size, seed=42)

    train_ds = split_ds["train"]
    validation_ds = split_ds["test"]

    run_finetuning("config_data_SFT", train_ds, validation_ds, tokenizer, configs)


if __name__ == "__main__":
    main()
