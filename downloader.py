from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys


model_name = "google/gemma-2-2b-it"
directory = model_name.split('/')[1]
model = AutoModelForCausalLM.from_pretrained(
    model_name
)
model.save_pretrained(directory)

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(directory)