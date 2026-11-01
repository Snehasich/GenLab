# Experiment 7: Implement using Python code to illustrate Parameter Efficient Fine-Tuning (PEFT) with LoRA using Hugging Face.

# Aim: Demonstrate parameter-efficient fine-tuning using LoRA.

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

name = "facebook/opt-125m"
token = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,       # predicts the next token
    r=8,            # rank
    lora_alpha=16,
    lora_dropout=0.1, 
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, config)           # Adds LoRA adapters to the pretrained model
model.print_trainable_parameters()

# model is LoRA-ready; train it with your labelled text dataset.
model.save_pretrained("opt-lora-adapter")
token.save_pretrained("opt-lora-adapter")