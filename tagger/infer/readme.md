# LLM-Powered Web Content Analyzer

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%2C%20Transformers%2C%20PEFT-orange)

This project uses large language models such as Qwen-32B to automate web content processing. It can fetch pages from a URL list, extract the main body text, and run a locally deployed PEFT-tuned model for analysis, summarization, information extraction, or other custom tasks.

## Setup and Installation

### 1. Prerequisites

* NVIDIA GPU. More than 64 GB of VRAM is recommended for running a 32B model with BF16 precision.
* Anaconda or Miniconda.
* CUDA Toolkit 11.8 or later.

### 2. Create and Activate a Conda Environment

Using Conda is recommended to keep project dependencies isolated. If you prefer not to create a new environment, install any missing packages from `requirements.txt` in your existing environment.

```bash
# Create an environment named web_an.
conda create -n web_an python=3.10 -y

# Activate the environment.
conda activate web_an
```

### 3. Install Core Dependencies

Install the PyTorch build that matches your CUDA version.

```bash
# Example: install PyTorch for CUDA 12.1.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

### 4. Run a Basic Test

The inference script expects a local model and adapter path.

1. Update the base model path in `infer_transformers.py`:

   ```python
   model_name = "Qwen/Qwen3-32B"
   ```

2. Update the LoRA adapter path in `infer_transformers.py`:

   ```python
   PeftModel.from_pretrained(model, "/root/autodl-tmp/qwen-qlora-split2", torch_dtype=torch.bfloat16)
   ```

3. Update the sample input text in `infer_transformers.py`:

   ```python
   message = "Morning all remember this"
   ```

### 5. Merge the Model

Use `merge_model.py` to merge the LoRA adapter into a standard Transformers-format model for deployment with tools such as vLLM. Configure the same base model path and LoRA adapter path used for inference, then set `output_dir` to the directory where the merged model should be written.
