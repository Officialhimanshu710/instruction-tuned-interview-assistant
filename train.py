import torch
import gc
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

torch.cuda.empty_cache()
gc.collect()

torch.backends.cuda.matmul.allow_tf32 = True
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

eos_token = "</s>"

def format_prompt(example):
    if example["context"]:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Context:\n{example['context']}\n\n### Response:\n{example['response']}{eos_token}"
    else:
        prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}{eos_token}"
    return {"text": prompt}

formatted_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

model_id = "mistralai/Mistral-7B-v0.1"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

max_memory_mapping = {0: "12GB", "cpu": "30GB"}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    max_memory=max_memory_mapping
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

training_args = SFTConfig(
    output_dir="./interview-assistant-results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,
    save_steps=50,
    dataset_text_field="text",
    max_length=512,
    fp16=False,
    bf16=False,
    optim="paged_adamw_8bit"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=training_args,
    processing_class=tokenizer
)

trainer.train()

trainer.model.save_pretrained("interview-assistant-lora")
tokenizer.save_pretrained("interview-assistant-lora")
