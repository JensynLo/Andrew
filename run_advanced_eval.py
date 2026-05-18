#!/usr/bin/env python3
"""
高级评估脚本：比较不同微调方法的效果
"""

import subprocess
import sys
import os
import argparse

def run_evaluation(args):
    """
    运行模型评估
    """
    cmd = [
        "python", "src/eval/eval_sft.py",
        "--base_model_path", args.base_model_path,
        "--sft_model_path", args.sft_model_path,
        "--eval_data_path", args.eval_data_path,
        "--output_path", args.output_path,
        "--device", args.device
    ]

    print("Running evaluation command:")
    print(" ".join(cmd))

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Evaluation completed successfully!")
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Compare different finetuned models")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Path to base model")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to SFT/DPO fine-tuned model")
    parser.add_argument("--eval_data_path", type=str, default="eval_data.json", help="Path to evaluation data")
    parser.add_argument("--output_path", type=str, default="eval_results.json", help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run evaluation on")

    args = parser.parse_args()

    print("Starting model evaluation...")
    print(f"Using device: {args.device}")
    print(f"Base model: {args.base_model_path}")
    print(f"SFT model: {args.sft_model_path}")

    run_evaluation(args)
    print(f"Check {args.output_path} for detailed results.")

if __name__ == "__main__":
    import torch  # 检查是否已安装torch

    main()