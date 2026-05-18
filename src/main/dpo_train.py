#!/usr/bin/env python
"""
DPO Training Script
Trains a model using Direct Preference Optimization on Terraria game assistant data
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.dpo_config import DPOConfig
from datasets import Dataset
from peft import LoraConfig
import json
import os
import argparse
from transformers import BitsAndBytesConfig


def format_dpo_example_with_chat_template(example, tokenizer):
    """
    Format a DPO example using the tokenizer's chat template to ensure consistency with SFT format
    """
    # Build messages in the format expected by apply_chat_template
    if "system" in example:
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
        # Format the prompt part (system + user) using chat template
        prompt_messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["instruction"]},
        ]
    else:
        messages = [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["chosen"]},
        ]
        # Format the prompt part (just user) using chat template
        prompt_messages = [
            {"role": "user", "content": example["instruction"]},
        ]

    # Get the formatted prompt that should match the SFT format
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def load_dpo_dataset(file_path: str, tokenizer) -> Dataset:
    """
    Load DPO dataset from JSONL file

    Args:
        file_path: Path to the DPO dataset file
        tokenizer: The tokenizer to use for applying chat templates

    Returns:
        Hugging Face Dataset with DPO format
    """
    raw_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                raw_data.append(data)

    # Process all examples using the tokenizer's chat template
    processed_examples = []
    for example in raw_data:
        processed_example = format_dpo_example_with_chat_template(example, tokenizer)
        processed_examples.append(processed_example)

    # Separate the fields
    prompts = [ex["prompt"] for ex in processed_examples]
    chosen_responses = [ex["chosen"] for ex in processed_examples]
    rejected_responses = [ex["rejected"] for ex in processed_examples]

    # Create dataset in the format expected by DPOTrainer
    dataset_dict = {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses,
    }

    return Dataset.from_dict(dataset_dict)


def prepare_model_and_tokenizer(model_path: str):
    """
    Load the SFT-trained model and tokenizer

    Args:
        model_path: Path to the SFT-trained model

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure quantization to save memory
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization to significantly reduce memory usage
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
        quantization_config=quantization_config,  # Use 4-bit quantization
        low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
    )

    return model, tokenizer


def train_dpo(
    model_path: str,
    dpo_data_path: str,
    output_dir: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 1,  # Reduce batch size to save memory
    gradient_accumulation_steps: int = 8,  # Increase gradient accumulation to compensate
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    warmup_steps: int = 100,
    save_steps: int = 500,
    eval_steps: int = 500,
    logging_steps: int = 10,
):
    """
    Train the model using Direct Preference Optimization

    Args:
        model_path: Path to the SFT-trained model to continue training from
        dpo_data_path: Path to the DPO training data
        output_dir: Directory to save the trained model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate for training
        beta: Beta parameter for DPO loss (controls divergence from reference model)
        warmup_steps: Number of warmup steps
        save_steps: Steps interval to save checkpoints
        eval_steps: Steps interval to evaluate
        logging_steps: Steps interval to log
    """
    # Load model and tokenizer
    model, tokenizer = prepare_model_and_tokenizer(model_path)

    # Load DPO dataset
    print(f"Loading DPO dataset from: {dpo_data_path}")
    dpo_dataset = load_dpo_dataset(dpo_data_path, tokenizer)

    print(f"Dataset loaded with {len(dpo_dataset)} examples")
    print(f"Sample from dataset:")
    sample = dpo_dataset[0]
    print(f"Prompt: {sample['prompt'][:200]}...")
    print(f"Chosen: {sample['chosen'][:200]}...")
    print(f"Rejected: {sample['rejected'][:200]}...")

    # Split dataset into train and eval if needed
    split_dataset = dpo_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Initialize DPO trainer
    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        bf16=True if torch.cuda.is_bf16_supported() else False,
        warmup_steps=warmup_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch",  # Use standard AdamW optimizer instead of bitsandbytes version
        report_to=[],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        # DPO-specific parameters
        beta=beta,
        max_length=512,  # Increased to accommodate longer prompts/responses
        # Additional memory optimizations
        remove_unused_columns=False,  # Keep all columns to prevent errors
        dataloader_pin_memory=False,  # Reduce memory usage
    )

    # Create a LoRA config for the DPO training
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    # 2. 初始化 DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Let TRL handle reference model internally with memory-efficient methods
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Updated parameter name for newer TRL versions
        peft_config=peft_config,  # Pass the LoRA config to the trainer
    )

    # Start training
    print("Starting DPO training...")
    dpo_trainer.train()

    # Save the final model
    print(f"Saving final model to {output_dir}")
    dpo_trainer.save_model(output_dir)

    # Save tokenizer to the same directory
    tokenizer.save_pretrained(output_dir)

    print("DPO training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train a model using Direct Preference Optimization"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the SFT-trained model to continue training from",
    )
    parser.add_argument(
        "--dpo_data_path",
        type=str,
        default="./output/dpo_training_data.jsonl",
        help="Path to the DPO training data file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/terraria_dpo_model",
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-7, help="Learning rate"
    )
    parser.add_argument(
        "--beta", type=float, default=0.1, help="Beta parameter for DPO loss"
    )
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Run DPO training
    train_dpo(
        model_path=args.model_path,
        dpo_data_path=args.dpo_data_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
