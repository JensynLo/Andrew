#!/usr/bin/env python
"""
DPO Training Script
Drives DPO training using configuration files and improved data logic
"""

import torch
import os
import argparse
import json
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from datasets import Dataset
from peft import LoraConfig
from ..utils import load_config


def format_dpo_example(example, tokenizer):
    """Format example with chat template for consistency"""
    if "system" in example and example["system"]:
        prompt_messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
        ]
    else:
        prompt_messages = [
            {"role": "user", "content": example["instruction"]},
        ]

    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def load_dpo_dataset(file_path: str, tokenizer) -> Dataset:
    raw_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))

    processed = [format_dpo_example(ex, tokenizer) for ex in raw_data]

    return Dataset.from_dict(
        {
            "prompt": [ex["prompt"] for ex in processed],
            "chosen": [ex["chosen"] for ex in processed],
            "rejected": [ex["rejected"] for ex in processed],
        }
    )


def train_dpo(configs):
    # Initialize wandb
    wandb.init(
        project=configs["wandb_project"],
        config=configs,
    )

    print(f"Loading model from: {configs['model']['model_id']}")
    tokenizer = AutoTokenizer.from_pretrained(configs["model"]["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        configs["model"]["model_id"],
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    dataset = load_dpo_dataset(configs["data"]["dpo_data_path"], tokenizer)
    split = dataset.train_test_split(test_size=configs["data"]["eval_split"])

    training_args = DPOConfig(
        output_dir=configs["training"]["output_dir"],
        num_train_epochs=configs["training"]["num_train_epochs"],
        per_device_train_batch_size=configs["training"]["per_device_train_batch_size"],
        per_device_eval_batch_size=configs["training"]["per_device_eval_batch_size"],
        gradient_accumulation_steps=configs["training"]["gradient_accumulation_steps"],
        learning_rate=configs["training"]["learning_rate"],
        beta=configs["training"]["beta"],
        max_length=configs["training"]["max_length"],
        warmup_steps=configs["training"]["warmup_steps"],
        save_strategy=configs["training"]["save_strategy"],
        save_steps=configs["training"]["save_steps"],
        save_total_limit=configs["training"]["save_total_limit"],
        eval_steps=configs["training"]["eval_steps"],
        logging_steps=configs["training"]["logging_steps"],
        logging_first_step=configs["training"]["logging_first_step"],
        loss_type=configs["training"]["loss_type"],
        eval_strategy="steps",
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=configs["training"]["remove_unused_columns"],
        gradient_checkpointing=True,
        report_to="wandb",
    )

    peft_config = LoraConfig(**configs["training"]["peft_config"])

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("Starting DPO training...")
    trainer.train()

    trainer.save_model(configs["training"]["output_dir"])
    tokenizer.save_pretrained(configs["training"]["output_dir"])
    print(f"Model saved to {configs['training']['output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="DPO Training Script")
    parser.add_argument("--config", type=str, default="configs/dpo_train_configs.json")
    parser.add_argument(
        "--build_data", action="store_true", help="Rebuild DPO dataset before training"
    )
    args = parser.parse_args()

    configs = load_config(args.config)

    if args.build_data:
        print("Rebuilding DPO dataset...")
        from ..dpo.data_generator import DPODataGenerator
        import pickle

        # Try to load graph
        G = None
        graph_path = "./output/knowledge_graph.pkl"  # Default path
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                G = pickle.load(f)

        generator = DPODataGenerator(G=G)
        sft_inputs = [
            "./output/blind_qa.jsonl",
            "./output/how_to_get_qa.jsonl",
            "./output/multiturn_dialogues.jsonl",
        ]
        generator.create_dataset(
            sft_inputs,
            configs["data"]["dpo_data_path"],
            max_samples=configs["data"]["max_samples"],
        )

    train_dpo(configs)


if __name__ == "__main__":
    main()
