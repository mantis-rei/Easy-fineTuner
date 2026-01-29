#!/usr/bin/env python3
"""
EASY FINE-TUNER - VPS Edition
==============================
Simple, reliable fine-tuning for any Linux VPS with GPU.

Usage:
    python train.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset "data.json"
    python train.py --model "mistralai/Mistral-7B-Instruct-v0.2" --dataset "data.json" --epochs 3

Supported dataset formats:
    - JSON: [{"text": "..."}, ...]
    - JSONL: {"text": "..."} per line
    - CSV: must have "text" column
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path

def check_dependencies():
    """Install required packages if missing."""
    required = ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"]
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print("✓ Dependencies installed!")

check_dependencies()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset, load_dataset

def get_gpu_info():
    """Detect GPU and return optimal settings."""
    if not torch.cuda.is_available():
        print("❌ No GPU detected! Training will be very slow.")
        return {"name": "CPU", "memory_gb": 8, "batch_size": 1, "max_seq": 512}
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    # Optimize settings based on GPU memory
    if gpu_mem >= 40:  # A100 40GB+
        settings = {"batch_size": 8, "max_seq": 2048, "lora_r": 32}
    elif gpu_mem >= 20:  # A10, 3090, 4090
        settings = {"batch_size": 4, "max_seq": 2048, "lora_r": 16}
    elif gpu_mem >= 10:  # T4, 3080
        settings = {"batch_size": 2, "max_seq": 1024, "lora_r": 8}
    else:  # Smaller GPUs
        settings = {"batch_size": 1, "max_seq": 512, "lora_r": 4}
    
    print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    return {"name": gpu_name, "memory_gb": gpu_mem, **settings}

def load_training_data(path: str) -> Dataset:
    """Load dataset from JSON, JSONL, or CSV."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    ext = path.suffix.lower()
    
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    
    elif ext == ".jsonl":
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    
    elif ext == ".csv":
        dataset = load_dataset("csv", data_files=str(path), split="train")
    
    else:
        raise ValueError(f"Unsupported format: {ext}. Use .json, .jsonl, or .csv")
    
    # Validate dataset has 'text' field
    if "text" not in dataset.column_names:
        raise ValueError("Dataset must have a 'text' column!")
    
    print(f"✓ Loaded {len(dataset)} training examples")
    return dataset

def train(
    model_name: str,
    dataset_path: str,
    output_dir: str = "./output",
    epochs: int = 3,
    learning_rate: float = 2e-4,
):
    """Main training function."""
    
    print("\n" + "=" * 50)
    print("  EASY FINE-TUNER")
    print("=" * 50)
    
    # Get GPU settings
    gpu = get_gpu_info()
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_path}")
    dataset = load_training_data(dataset_path)
    
    # Configure quantization
    print(f"\nLoading model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    print("✓ Model loaded!")
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=gpu["lora_r"],
        lora_alpha=gpu["lora_r"] * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ LoRA applied: {trainable:,} trainable params ({100*trainable/total:.2f}%)")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=gpu["batch_size"],
        gradient_accumulation_steps=max(1, 16 // gpu["batch_size"]),
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=gpu["max_seq"],
        packing=False,
    )
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Train!
    print(f"\n🚀 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {gpu['batch_size']}")
    print(f"   Max sequence: {gpu['max_seq']}")
    print("=" * 50 + "\n")
    
    trainer.train()
    
    print("\n✓ Training complete!")
    
    # Save model
    print("\nSaving model...")
    
    # Save LoRA adapters
    lora_path = f"{output_dir}/lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"✓ LoRA adapters saved: {lora_path}")
    
    # Merge and save full model
    merged_path = f"{output_dir}/merged"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"✓ Merged model saved: {merged_path}")
    
    print("\n" + "=" * 50)
    print("  TRAINING COMPLETE!")
    print("=" * 50)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - LoRA adapters: {lora_path}/")
    print(f"  - Full model: {merged_path}/")
    print("\nTo use your model:")
    print(f'  from transformers import AutoModelForCausalLM, AutoTokenizer')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{merged_path}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{merged_path}")')

def main():
    parser = argparse.ArgumentParser(
        description="Easy Fine-Tuner - Train any HuggingFace model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --dataset "data.json"
  python train.py --model "mistralai/Mistral-7B-Instruct-v0.2" --dataset "data.json" --epochs 5
  python train.py --model "Qwen/Qwen2-1.5B-Instruct" --dataset "data.csv" --output "./my-model"
        """
    )
    
    parser.add_argument("--model", "-m", required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", "-d", required=True, help="Path to training data (JSON/JSONL/CSV)")
    parser.add_argument("--output", "-o", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    
    args = parser.parse_args()
    
    train(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        learning_rate=args.lr,
    )

if __name__ == "__main__":
    main()
