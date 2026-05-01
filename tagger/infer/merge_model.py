from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

output_dir = "/root/autodl-tmp/qwen-sft"

model_name = "/root/autodl-tmp"# !!!!!! change with Qwen/Qwen3-32B
from get_sample_template import get_sample
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/qwen-qlora-split5", torch_dtype=torch.bfloat16)
model.merge_and_unload()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)