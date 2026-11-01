# Experiment 5: Build a text generation model using HuggingFace Transformers and a pretrained GPT model.

# Aim: Generate text using pretrained GPT-2.

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = "In the near future, artificial intelligence in education will"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.generate(
    **inputs, max_new_tokens=60, do_sample=True,
    temperature=0.8, top_p=0.9, num_return_sequences=3,
    pad_token_id=tokenizer.eos_token_id
)

for i, output in enumerate(outputs, 1):
    print(f"\n=== Generation {i} ===")
    print(tokenizer.decode(output, skip_special_tokens=True))

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))     # converts it into readable text