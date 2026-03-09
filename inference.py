import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-v0.1"
adapter_path = "/content/interview-assistant-lora" # Note: Updated path for Colab

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

question = "Explain the difference between a list and a tuple in Python. How should a candidate answer this in a technical interview?"

prompt = f"### Instruction:\n{question}\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating answer...\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("----- MODEL RESPONSE -----\n")
print(response)
