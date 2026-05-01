from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
model_name = "/root/autodl-tmp"# !!!!!! change with Qwen/Qwen3-32B
from get_sample_template import get_sample
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/qwen-qlora-split2", torch_dtype=torch.bfloat16)
model.merge_and_unload()

#please set the message you want to test here
message = "Morning all remember this project is fantastic as you investment is on a 24 cycle not locked up https://bnbhour.com/?ref=0x7D749Dc4023663756B5Febc5be88d5A3eeB1078E"
input_message = get_sample(message)
inputs = tokenizer.apply_chat_template(
        input_message, 
        tokenize=True, 
        add_generation_prompt=True, # 添加 assistant 提示符以精确定位
        return_tensors="pt"
    ).cuda()
# print(inputs)
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=4096,  # 设置一个合理的长度，防止生成过长
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=False, # 使用确定性生成，关闭采样
    )
response_ids = outputs[0][len(inputs[0]):]
generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)
print("Generated Response: ", generated_text.replace("<think>","").replace("</think>","").strip())