from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/Users/lily/llm-model-data/chatglm2-6b" # 用你的模型的地址
# 增加trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)

text = "say"
inputs = tokenizer(text, return_tensors="pt")
print(f"inputs:{inputs}")

#预测下一个词
logits = model.forward(inputs)
print("Logits Shape:", logits.logits.shape)
print(f"logits:{logits.logits}")