# Experiment 9: Implement Zero-Shot vs Few-Shot Prompting

# Aim: To compare how GPT-2 classifies a review using zero-shot, one-shot, and three-shot prompting.

from transformers import pipeline

gen = pipeline("text-generation", model="gpt2", max_new_tokens=10, pad_token_id=50256)

review = "The keyboard feels mushy and keys register inconsistently."

prompts = {
    "Zero-shot": f"Classify as Positive or Negative. Review: {review}\nSentiment:",
    "One-shot": f"Review: Excellent quality. Sentiment: Positive\nReview: {review}\nSentiment:",
    "Three-shot": f"Review: Great product. Sentiment: Positive\nReview: Poor battery. Sentiment: Negative\nReview: Fast delivery. Sentiment: Positive\nReview: {review}\nSentiment:"
}
    
for label, prompt in prompts.items():
    result = gen(prompt, pad_token_id=50256)[0]["generated_text"]
    print(label, "->", result.split("Sentiment:")[-1].strip().split()[0])


'''
    split("Sentiment:"): separates the text at Sentiment:.
    [-1]: takes the last part.
    strip(): removes extra spaces.
    split()[0]: takes the first generated word.
'''