import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import json
import argparse
import os
from typing import List, Dict, Tuple
from peft import PeftModel


def calculate_perplexity(
    model, tokenizer, texts: List[str], device: str = "cuda"
) -> float:
    """
    Calculate perplexity for a list of texts
    """
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask").to(device) if "attention_mask" in inputs else None

            outputs = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))

    return perplexity.item()


def calculate_cross_entropy_loss(
    model, tokenizer, texts: List[str], device: str = "cuda"
) -> float:
    """
    Calculate average cross entropy loss for texts
    """
    model.eval()
    total_loss = 0
    num_samples = len(texts)

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask").to(device) if "attention_mask" in inputs else None

            outputs = model(input_ids=input_ids, labels=input_ids, attention_mask=attention_mask)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / num_samples


def calculate_generation_metrics(
    base_model, sft_model, tokenizer, eval_prompts: List[str], device: str = "cuda"
) -> Dict:
    """
    Compare generation metrics between base and SFT models
    """
    base_model.eval()
    sft_model.eval()

    base_generated_texts = []
    sft_generated_texts = []

    with torch.no_grad():
        for prompt in eval_prompts:
            # Tokenize prompt
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=False,
                max_length=512,
            )
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask").to(device) if "attention_mask" in inputs else None

            # Generate from base model
            base_outputs = base_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            base_text = tokenizer.decode(
                base_outputs[0][input_ids.size(1) :], skip_special_tokens=True
            )
            base_generated_texts.append(base_text)

            # Generate from SFT model
            sft_outputs = sft_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            sft_text = tokenizer.decode(
                sft_outputs[0][input_ids.size(1) :], skip_special_tokens=True
            )
            sft_generated_texts.append(sft_text)

    # Here we can add more sophisticated metrics like BLEU, ROUGE scores
    # For now, just return basic generation comparison
    return {
        "num_prompts": len(eval_prompts),
        "base_generated_examples": base_generated_texts[:3],  # First 3 examples
        "sft_generated_examples": sft_generated_texts[:3],  # First 3 examples
    }


def evaluate_models(
    base_model_path: str,
    sft_model_path: str,
    eval_data_path: str,
    output_path: str,
    device: str = "cuda",
):
    """
    Main evaluation function to compare base and SFT models
    """
    print(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16
    )
    base_model.to(device)  # type:ignore

    print(f"Loading SFT model from {sft_model_path}")
    # Check if sft_model_path is a LoRA adapter
    if os.path.exists(os.path.join(sft_model_path, "adapter_config.json")):
        print("Detected LoRA adapter, loading base model first and applying LoRA")
        # Load base model first, then apply LoRA adapter
        base_for_lora = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.float16
        )
        sft_model = PeftModel.from_pretrained(base_for_lora, sft_model_path)
        sft_model = sft_model.merge_and_unload()  # Merge LoRA weights for evaluation
    else:
        # Standard model loading
        sft_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path, torch_dtype=torch.float16
        )
    sft_model.to(device)

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 设置clean_up_tokenization_spaces以避免警告
    if hasattr(tokenizer, 'clean_up_tokenization_spaces'):
        tokenizer.clean_up_tokenization_spaces = False

    # Load evaluation data
    print(f"Loading evaluation data from {eval_data_path}")
    with open(eval_data_path, "r", encoding="utf-8") as f:
        if eval_data_path.endswith(".jsonl"):
            eval_data = [json.loads(line) for line in f]
        elif eval_data_path.endswith(".json"):
            eval_data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use .json or .jsonl")

    # Extract texts for evaluation
    if isinstance(eval_data[0], dict):
        # Assume the dataset has 'text' or 'input' and 'output' keys
        if "text" in eval_data[0]:
            eval_texts = [item["text"] for item in eval_data]
        elif "input" in eval_data[0] and "output" in eval_data[0]:
            eval_texts = [item["input"] + item["output"] for item in eval_data]
        elif "prompt" in eval_data[0] and "response" in eval_data[0]:
            eval_texts = [item["prompt"] + item["response"] for item in eval_data]
        else:
            eval_texts = [str(item) for item in eval_data]
    else:
        eval_texts = [str(item) for item in eval_data]

    # Extract prompts for generation comparison
    eval_prompts = []
    if "input" in eval_data[0]:
        eval_prompts = [
            item["input"] for item in eval_data[:10]
        ]  # Use first 10 as prompts
    elif "prompt" in eval_data[0]:
        eval_prompts = [
            item["prompt"] for item in eval_data[:10]
        ]  # Use first 10 as prompts
    else:
        # If no explicit prompts, use first part of texts
        eval_prompts = [text[: min(len(text) // 2, 100)] for text in eval_texts[:10]]

    print("Calculating metrics...")

    # Calculate cross entropy losses
    base_ce_loss = calculate_cross_entropy_loss(
        base_model, tokenizer, eval_texts, device
    )
    sft_ce_loss = calculate_cross_entropy_loss(sft_model, tokenizer, eval_texts, device)

    # Calculate perplexities
    base_perplexity = calculate_perplexity(base_model, tokenizer, eval_texts, device)
    sft_perplexity = calculate_perplexity(sft_model, tokenizer, eval_texts, device)

    # Calculate generation metrics
    gen_metrics = calculate_generation_metrics(
        base_model, sft_model, tokenizer, eval_prompts, device
    )

    # Compile results
    results = {
        "model_comparison": {
            "base_model_path": base_model_path,
            "sft_model_path": sft_model_path,
            "eval_data_path": eval_data_path,
        },
        "metrics": {
            "cross_entropy_loss": {
                "base_model": base_ce_loss,
                "sft_model": sft_ce_loss,
                "improvement": base_ce_loss
                - sft_ce_loss,  # Positive means SFT is better
            },
            "perplexity": {
                "base_model": base_perplexity,
                "sft_model": sft_perplexity,
                "improvement": base_perplexity
                - sft_perplexity,  # Positive means SFT is better
            },
        },
        "generation_comparison": gen_metrics,
        "summary": {
            "loss_improvement_percentage": ((base_ce_loss - sft_ce_loss) / base_ce_loss)
            * 100
            if base_ce_loss != 0
            else 0,
            "perplexity_improvement_percentage": (
                (base_perplexity - sft_perplexity) / base_perplexity
            )
            * 100
            if base_perplexity != 0
            else 0,
        },
    }

    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Base Model CE Loss: {base_ce_loss:.4f}")
    print(f"SFT Model CE Loss: {sft_ce_loss:.4f}")
    print(
        f"Loss Improvement: {results['metrics']['cross_entropy_loss']['improvement']:.4f} ({results['summary']['loss_improvement_percentage']:.2f}%)"
    )
    print("-" * 30)
    print(f"Base Model Perplexity: {base_perplexity:.4f}")
    print(f"SFT Model Perplexity: {sft_perplexity:.4f}")
    print(
        f"Perplexity Improvement: {results['metrics']['perplexity']['improvement']:.4f} ({results['summary']['perplexity_improvement_percentage']:.2f}%)"
    )
    print("=" * 50)

    # Save results
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SFT model against base model"
    )
    parser.add_argument(
        "--base_model_path", type=str, required=True, help="Path to base model"
    )
    parser.add_argument(
        "--sft_model_path", type=str, required=True, help="Path to SFT fine-tuned model"
    )
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to evaluation data (JSON/JSONL)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./eval_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run evaluation on",
    )

    args = parser.parse_args()

    evaluate_models(
        base_model_path=args.base_model_path,
        sft_model_path=args.sft_model_path,
        eval_data_path=args.eval_data_path,
        output_path=args.output_path,
        device=args.device,
    )
