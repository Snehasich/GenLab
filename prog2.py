# Experiment 2: Develop a python code to illustrate attention mechanism visualization.

# Aim: Visualize how BERT attends to tokens in a sentence.

import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()

sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    attention = model(**inputs).attentions

sns.heatmap(attention[5][0, 3].numpy(),
            xticklabels=tokens, yticklabels=tokens, cmap="Blues")
plt.title("Attention - Layer 6, Head 4")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150)
plt.show()
