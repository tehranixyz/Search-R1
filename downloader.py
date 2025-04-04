from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct"
)
model.save_pretrained('Qwen2-0.5B-Instruct')

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer.save_pretrained('Qwen2-0.5B-Instruct')