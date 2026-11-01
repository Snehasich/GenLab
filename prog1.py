# 1. Develop a simple Transformer for Text Classification using HuggingFace

# Aim: Identify whether text is positive or negative.

from transformers import pipeline
 
clf = pipeline("text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english")
for text in ["AI is amazing!", "This result is disappointing."]:
    print(text, "->", clf(text)[0])
