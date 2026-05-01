# 项目名称

## 目录
- [数据集处理](#数据集处理)
- [训练](#训练)
- [测试](#测试)

## 数据集处理
(请在此处添加您的数据集处理说明)
数据集处理文件
为了丰富和增强原始数据集，我们为每个样本提供了两种额外的信息获取方式。详见（sft-gpt/infer/get_sample_template.py）具体处理流程如下：

1.  **网址内容提取 (URL Content Extraction):**
    如果样本的文本中包含任何网址 (URL)，程序会自动尝试访问这些网址，并抓取其网页内容作为补充信息。

2.  **网页搜索增强 (Web Search Augmentation):**
    如果样本包含网址，且网址内容不可提取，程序会执行网页搜索。并将其整合到数据中。

所有数据均在sft-gpt/data中。


## 训练
本项目仅仅包含主流transformers模型的训练：

### . 微调 Qwen, Llama 等模型
- **文件路径:** `sft-gpt/train/train_qlora.py`
- **说明:** 此脚本用于微调如 Qwen, Llama 等其他模型。
- **环境配置:**
  请确保您安装了最新版本的 PyTorch, Transformers 和 BitsAndBytes。您可以使用以下命令进行安装：
  ```bash
  pip install --upgrade torch transformers bitsandbytes
  ```
  **注意:**
  * `bitsandbytes` 目前主要支持 Linux 发行版，Windows 支持尚不完善。
  * 请根据您的硬件（如 NVIDIA CUDA 版本）和软件需求，参考官方文档以获取更详细的安装说明。
- **代码修改:**
    请修改训练代码中的：
  ```python
    model_name = "/root/autodl-tmp"
    data_path = "/root/train_dataset_all.json"
    output_dir = "/root/autodl-tmp/qwen-qlora-split5"
  ```
  其中model_name作为模型，可以是huggingface上的模型库，也可以是本地下载的模型。
---

## 测试
测试流程分为两步，分别用于获取初步结果和最终结果。

### 1. 初步测试 
- **文件路径:** `sft-gpt/test.ipynb`
- **说明:** 运行此 Jupyter Notebook 可获得模型（ Qwen, Llama等）的初步测试结果（完全匹配度）。并保存在evaluation_results.json文件
- **代码修改:** 您需要修改
```python
test_data_path = "/root/test_prompts (5).json"
model_name = "/root/autodl-tmp"
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/qwen-qlora-split2", torch_dtype=torch.bfloat16)
```
以上文件

### 2. 最终结果生成
- **文件路径:** `sft-gpt/utils/test.py`
- **说明:** 在完成初步测试后，执行此脚本以处理数据并获得最终的评估结果。请将evaluation_results.json文件的路径替换为代码中input_file 的路径，并且执行代码即可。

