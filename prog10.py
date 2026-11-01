# Experiment 10: Implement the concept of Chain-of-Thought (CoT) Prompting with Self-Consistency

# teaches GPT-2 to solve a mathematical question using an example and generate multiple answers.

import re                                   # find the numerical answer in a text
from collections import Counter             # counts repeated answers.
from transformers import pipeline           # loads and uses the GPT-2 model.

gen = pipeline("text-generation", model="gpt2", max_new_tokens=50,
               pad_token_id=50256, do_sample=True)

direct = "Train travels at 60 km/h for 2 hours. Distance?"

cot = """Q: Car travels at 40 km/h for 3 hours.
        A: Distance = 40 x 3 = 120 km. Answer: 120 km.
        Q: Train travels at 60 km/h for 2 hours.
        A: Distance = 60 x 2 = 120 km. Answer: 120 km.
        Answer:"""

print("Direct:", gen(direct)[0]["generated_text"])

answers = []
for i in range(5):
    out = gen(cot, max_new_tokens=100)[0]["generated_text"]     # Sends the CoT prompt to GPT-2
    m = re.search(r"Answer:\s*([\d.]+)\s*km", out)              # Searches for a number after the word Answer:.
    if m:
        answers.append(m.group(1))

print("Answers:", answers)
if answers:
    print("Majority:", Counter(answers).most_common(1)[0][0], "km")


"""
    Counter(answers) – counts each answer.
    .most_common(1) – finds the most repeated answer.
    [0][0] – gets that answer.
"""