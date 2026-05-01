# train_qlora.py (V3 - 正确版)

# train.py

import torch
import os
import json
import random
from functools import partial
# from unsloth import FastLanguageModel
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    Mxfp4Config


)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 全局变量 ---
# 从您的任务描述中复制 system_instructions
system_instructions = r"""<<BEGIN-CODEBOOK-RULES>> ... <<END-CODEBOOK-RULES>>"""

# --- 1. 数据集类 ---
class SFTDataset(Dataset):
    """用于监督式微调的简单数据集类"""
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

# --- 2. 数据整理函数 (关键!) ---
def collate_fn(batch: list, tokenizer: AutoTokenizer) -> dict:
    """
    数据整理函数，将我们的对话数据正确编码为模型输入和标签。
    只计算 assistant 回答部分的损失。
    """
    # 1. 为批次中的每个样本构建聊天消息列表
    messages_batch = []
    for item in batch:
        messages = item['message']
        messages_batch.append(messages)

    # 2. 使用 apply_chat_template 将对话格式化为单个字符串
    #    这里我们不立即编码，因为我们需要找到标签的起始位置
    full_texts_without_tokenization = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False) 
        for m in messages_batch
    ]
    
    # 3. 对完整文本进行分词
    inputs = tokenizer(
        full_texts_without_tokenization,
        padding="longest",
        truncation=True,
        max_length=4096, # 与 SFTTrainer 版本保持一致
        return_tensors="pt"
    )

    # 4. 创建标签，核心逻辑：只计算 assistant 回答部分的损失
    labels = inputs.input_ids.clone()
    print(labels.shape)
    # 找到每个样本中 assistant 回答的起始位置，并屏蔽之前的部分
    for i, messages in enumerate(messages_batch):
        # 只包含 system 和 user 的部分，用于定位 prompt 的结束位置
        prompt_messages = messages[:-1]
        
        # apply_chat_template 也能处理不完整的对话，并正确地添加 assistant 提示符
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True # 添加 assistant 提示符以精确定位
        )
        
        # 编码以获取长度
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_start_idx = len(prompt_ids)

        # 将 prompt 部分的标签设置为 -100
        labels[i, :response_start_idx] = -100

    # 将 padding token 对应的标签也设置为 -100
    labels[inputs.attention_mask == 0] = -100

    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": labels
    }

def train():
    import argparse

    parser = argparse.ArgumentParser(description="QLoRA SFT for the codebook tagger.")
    parser.add_argument("--model_name", required=True, help="HF model name or local path (e.g., Qwen3-32B).")
    parser.add_argument("--data_path", required=True, help="Path to train_dataset_all.json (chat-format).")
    parser.add_argument("--output_dir", required=True, help="Output directory for the LoRA adapter + tokenizer.")
    args = parser.parse_args()

    # --- 模型和数据路径 ---
    model_name = args.model_name
    data_path = args.data_path
    output_dir = args.output_dir

    # --- 设置随机种子 ---
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # --- QLoRA 配置 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # --- 加载 Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # 确认 chat_template 已加载
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer must have a chat_template.jinja file in its directory.")
    max_memory = {0: "50GiB", 1: "50GiB",'cpu': '300GiB'}
    # --- 加载模型 ---
    quantization_config = Mxfp4Config(dequantize=True)
    print("load model")
    # max_memory = {0: "60GiB", 1: "75GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        # attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        attn_implementation="flash_attention_2",
        use_cache=False,
        # trust_remote_code=True,
        # max_memory=max_memory,
        # use_flash_attention_2=True,
    )

    # --- PEFT (LoRA) 配置 ---
    peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",
    # lora_dropout=0.05, 
    # target_parameters=[
    #     "7.mlp.experts.gate_up_proj",
    #     "7.mlp.experts.down_proj",
    #     "15.mlp.experts.gate_up_proj",
    #     "15.mlp.experts.down_proj",
    #     "23.mlp.experts.gate_up_proj",
    #     "23.mlp.experts.down_proj",
    # ],
)
    # model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    # --- 加载数据集 ---
    dataset = SFTDataset(data_path)

    # --- 配置训练参数 ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=5,
        num_train_epochs=2,
        # max_steps=50,
        save_steps=25,
        save_total_limit=2,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        # deepspeed=ds_config,
        # optim="paged_adamw_8bit",
    )

    # --- 初始化 Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
    )

    # --- 开始训练 ---
    print("Starting training with custom collate_fn...")
    trainer.train()

    # --- 保存最终模型 ---
    print("Training complete. Saving final LoRA adapter.")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Final LoRA adapter and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    train()
