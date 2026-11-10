# QIA: Quantum-Inspired Adaptation

**QIA** is a high-efficiency fine-tuning technique for Large Language Models (LLMs). It uses complex-valued phase interference to suppress hallucinations and amplify correct pathways.

## Installation
```bash
pip install git+[https://github.com/Muhammadamin0334/Quantum-Inspired-Adaptation.git](https://github.com/Muhammadamin0334/Quantum-Inspired-Adaptation.git)
```
Quickstart & Usage
Below is a complete example showing how to load a model, inject QIA adapters, prepare data, and run fine-tuning using QIATrainer.

import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    DataCollatorForSeq2Seq
)
from datasets import Dataset

# Import QIA framework modules
from qira import inject_qira, QIRATrainer

# 1. Load Base Model and Tokenizer
MODEL_ID = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# 2. Inject QIA Reservoir Adapters
# Freezes base model and injects phase-mixing adapters into attention layers
model = inject_qira(
    model, 
    target_modules=["c_attn"],  # Use ["q_proj", "v_proj"] for Llama/Mistral models
    reservoir_dim=128, 
    alpha=0.03
)

# 3. Prepare Dataset
train_data = [
    {
        "instruction": "What is QIA?", 
        "output": "QIA is a quantum iterative adaptation technique for efficient LLM fine-tuning and hallucination suppression."
    },
    {
        "instruction": "How does QIA work?", 
        "output": "It utilizes phase-shift parameters and trigonometric wave mixing to amplify correct pathways and minimize output noise."
    }
]

def format_data(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    encoded = tokenizer(prompt, truncation=True, max_length=128, padding="max_length")
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded

train_dataset = Dataset.from_list(train_data).map(format_data, remove_columns=["instruction", "output"])
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8)

# 4. Configure Training Arguments & Trainer
training_args = TrainingArguments(
    output_dir="./qia_training_results",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=3e-4,
    logging_steps=2,
    save_strategy="no",
    report_to="none"
)

trainer = QIRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    phase_reg_weight=1e-3  # Regularization penalty for phase stability
)

# 5. Train
trainer.train()

# 6. Test Generation
model.eval()
test_prompt = "### Instruction:\nWhat is QIA?\n\n### Response:\n"
inputs = tokenizer(test_prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=40, 
        temperature=0.7, 
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
