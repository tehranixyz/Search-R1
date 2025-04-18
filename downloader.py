from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"
save_path = model_name.split("/")[-1]
model = AutoModelForCausalLM.from_pretrained(
    model_name
)
model.save_pretrained(save_path)

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)