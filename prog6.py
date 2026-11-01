# Experiment 6: Experiment with GPT-based summarization or Q&A

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def generate(prompt, n=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=n,
        do_sample=True, temperature=0.7, top_p=0.9,
        pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def summarize(text, detail):
    prompt = ("Summarize in 2-3 sentences: " if detail == "short"
              else "Write a detailed summary: ") + text
    return generate(prompt, 150)

def answer(context, question):
    return generate("Context: " + context +
                    "\nQuestion: " + question + "\nAnswer:", 80)

text = "AI helps education through personalized learning, tutoring and automated grading. It also raises privacy and bias concerns."

print("Short Summary:", summarize(text, "short"))
print("Detailed Summary:", summarize(text, "detailed"))

context = "GPT generates text by predicting the next token one at a time."
question = "How does GPT generate text?"

print("Answer:", answer(context, question))