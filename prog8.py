# Experiment 8: Python code to illustrate Prompt Engineering Basics (Prompting, Formatting, Best Practices)

# Using Hugging Face text-generation pipeline

from transformers import pipeline

gen = pipeline("text-generation", model="gpt2", max_new_tokens=120, pad_token_id=50256)

prompts = {
    "Baseline": "What is machine learning?",
    "Role": "Explain machine learning as a professor.",
    "Format": "Explain machine learning in 3 bullet points.",
    "JSON": "Explain machine learning in JSON format."
}

for label, prompt in prompts.items():
    print(f"\n=== {label} ===")
    print(gen(prompt)[0]["generated_text"])