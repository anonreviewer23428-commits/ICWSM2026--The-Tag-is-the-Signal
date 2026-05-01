# LLM-Powered Web Content Analyzer (基于LLM的网页内容分析器)

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%2C%20Transformers%2C%20PEFT-orange)

这是一个利用大型语言模型（LLM）自动化处理网页内容的强大工具。项目能够抓取指定URL列表的网页，智能提取核心正文，并调用本地部署的、经过PEFT微调的Qwen-32B模型进行深度分析、摘要、信息提取或其他自定义任务。

This project leverages the power of Large Language Models (LLMs) like Qwen-32B to automate the process of fetching, parsing, and analyzing content from web pages. It is designed to take a list of URLs, extract the main body content intelligently, and then use a locally deployed, PEFT-tuned model to perform deep analysis, summarization, information extraction, or other custom tasks.


## 环境准备与安装 (Setup and Installation)

### 1. 先决条件 (Prerequisites)

*   NVIDIA GPU (推荐 VRAM > 64GB 用于BF16精度的32B模型)
*   Anaconda 或 Miniconda
*   CUDA Toolkit 11.8 或更高版本


### 2. 创建并激活 Conda 环境(注：可不新建conda环境，运行时缺少那个库安装即可，在requirements.txt文件)

我们强烈建议使用Conda来管理项目依赖，以避免版本冲突。

```bash
# 创建一个名为 'web_an' 的环境
conda create -n web_an python=3.10 -y

# 激活环境
conda activate web_an
```

### 3. 安装核心依赖

首先，安装与您CUDA版本匹配的PyTorch。

```bash
# 示例：安装兼容 CUDA 12.1 的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


使用pip安装所有依赖：

```bash
pip install -r requirements.txt
```

### 4. 初步测试

本项目需要加载一个本地的大型语言模型。

1.  **修改模型路径**:
    打开您的主代码文件（`infer_transformers.py`），找到以下这行并修改为您本地模型的实际路径：
    ```python
    # 将 "/root/autodl-tmp" 修改为你的模型文件夹路径
    model_name = "Qwen/Qwen3-32B" 
    ```
2.  **修改lora模块路径**:
    打开您的主代码文件（`infer_transformers.py`），找到以下这行
    将 
    ```python 
    PeftModel.from_pretrained(model, "/root/autodl-tmp/qwen-qlora-split2", torch_dtype=torch.bfloat16)
    ```
    中的路径修改为你的lora模块文件夹路径
3. **修改lora模块路径**:
    打开您的主代码文件（`infer_transformers.py`），找到以下这行，将 
    ```python 
    message = "Morning all remember this"
    ```
    中的路径修改为您所需要测试的文本

---

### 5. 模型合并

将lora模型合并为transformers的官方格式模型，以适用于vllm等部署。所有步骤与4相同，对于merge_model.py文件。

1.  **修改模型路径**
2.  **修改lora模块路径**
2.  **修改输出路径**:
output_dir为您想要输出的路径。
    
---