# Easy Fine-Tuner

Simple, reliable fine-tuning for any HuggingFace model on Linux VPS with GPU.

## Features

- 🚀 Auto-detects GPU and optimizes settings
- 📝 Converts plain text to training datasets
- 🔄 Batch process multiple text files
- 💾 Saves both LoRA adapters and merged model
- ⚡ Works with any HuggingFace model

## Quick Start

### 1. Clone to your VPS

```bash
git clone https://github.com/YOUR_USERNAME/easy-finetuner.git
cd easy-finetuner
```

### 2. Setup (first time only)

```bash
bash setup.sh
source venv/bin/activate
```

### 3. Add your training data

Put `.txt` files in the `data/` folder, then:

```bash
python batch_convert.py
```

### 4. Train!

```bash
python train.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset dataset.json
```

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│  1. CREATE TEXT FILES                                   │
│     Write your knowledge in .txt files                  │
│     (Q&A format, headers, paragraphs, etc.)            │
│                                                         │
│  2. PUT IN data/ FOLDER                                 │
│     Copy your .txt files to data/                       │
│                                                         │
│  3. CONVERT TO DATASET                                  │
│     python batch_convert.py                             │
│     → Creates dataset.json                              │
│                                                         │
│  4. TRAIN MODEL                                         │
│     python train.py -m MODEL -d dataset.json            │
│     → Creates output/merged/ (your model!)              │
│                                                         │
│  5. USE YOUR MODEL                                      │
│     Download output/merged/ folder                      │
│     Load with transformers or convert to GGUF           │
└─────────────────────────────────────────────────────────┘
```

---

## Data Formats

### Q&A Format (Best for FAQs, tutorials)
```
Q: What is machine learning?
A: Machine learning is a subset of AI where computers learn from data...

Q: How does it work?
A: It works by finding patterns in training data...
```

### Headers Format (Best for topics, documentation)
```
### Machine Learning Basics
Machine learning is a field of AI that enables computers to learn...

### Types of ML
There are three main types: supervised, unsupervised, and reinforcement...
```

### Numbered Format (Best for step-by-step, lists)
```
1. Introduction to ML
Machine learning enables computers to learn from data...

2. Getting Started
First, install the required libraries...
```

### Paragraphs Format (Best for articles, books)
```
Machine learning has revolutionized how we approach problems...

Deep learning is a subset of machine learning using neural networks...
```

---

## Commands

### Convert single file
```bash
python convert.py input.txt -o dataset.json
python convert.py input.txt -o dataset.json --format qa
python convert.py input.txt -o dataset.json --system "You are an expert."
```

### Batch convert all files
```bash
python batch_convert.py                  # Uses data/ folder
python batch_convert.py -i myfolder/     # Custom folder
python batch_convert.py -s "You are a trading expert."
```

### Train model
```bash
python train.py -m TinyLlama/TinyLlama-1.1B-Chat-v1.0 -d dataset.json
python train.py -m mistralai/Mistral-7B-Instruct-v0.2 -d dataset.json -e 5
python train.py -m Qwen/Qwen2-1.5B-Instruct -d dataset.json -o mymodel
```

---

## Recommended Models

| Model | Size | Quality | Use Case |
|-------|------|---------|----------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1B | Good | Fast testing |
| `Qwen/Qwen2-1.5B-Instruct` | 1.5B | Great | Balanced |
| `microsoft/phi-2` | 2.7B | Great | Reasoning |
| `mistralai/Mistral-7B-Instruct-v0.2` | 7B | Excellent | Production |

---

## GPU Requirements

The script auto-detects your GPU and adjusts settings:

| GPU Memory | Batch Size | Max Sequence | Models |
|------------|------------|--------------|--------|
| 40GB+ | 8 | 2048 | Any |
| 20-40GB | 4 | 2048 | Up to 13B |
| 10-20GB | 2 | 1024 | Up to 7B |
| <10GB | 1 | 512 | Up to 3B |

---

## Output

After training, find your model in `output/`:

```
output/
├── checkpoints/     # Training checkpoints
├── lora/            # LoRA adapters (small, efficient)
└── merged/          # Full model (ready to use)
```

### Using your model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("output/merged")
tokenizer = AutoTokenizer.from_pretrained("output/merged")

# Generate text
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## VPS Providers

| Provider | GPU | Cost | Best For |
|----------|-----|------|----------|
| [Vast.ai](https://vast.ai) | Various | $0.10-0.50/hr | Cheapest |
| [RunPod](https://runpod.io) | A100/4090 | $0.20-1.00/hr | Reliable |
| [Lambda](https://lambdalabs.com) | A100 | $1.10/hr | Enterprise |

---

## License

MIT License - Use freely!
