from transformers import AutoModel

model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with your model

try:
    model = AutoModel.from_pretrained(model_name)
    print(f"Model '{model_name}' is already downloaded and loaded successfully.")
except:
    print(f"Model '{model_name}' is NOT downloaded or cannot be loaded.")
