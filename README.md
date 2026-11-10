<div align="center">

# QIA: Quasi-Orthogonal Iterative Adaptation

<p align="center">
  <b>A high-efficiency, lightweight fine-tuning framework for Large Language Models powered by reservoir computing and trigonometric wave mixing.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status">
</p>

</div>

---

## 📖 Overview

**QIA (Quasi-Orthogonal Iterative Adaptation)** is an advanced, lightweight fine-tuning technique designed for Large Language Models (LLMs). By freezing base model weights and leveraging structured phase-shift parameters alongside trigonometric wave mixing, QIA stabilizes output representations, suppresses hallucinations, and achieves high training efficiency with minimal computational overhead.

---

## ✨ Key Features

* **Parameter-Efficient Training:** Freezes base model parameters and isolates updates to lightweight adapter layers and dual-component readout matrices.
* **Phase-Shift Parameterization:** Employs dynamic phase shifts and trigonometric wave mixing to amplify correct semantic trajectories and destructively interfere with noise.
* **Built-In Wave Regularization:** Includes a custom `QIATrainer` featuring L2 regularization over phase-shift parameters to maintain mathematical stability during gradient descent.
* **Seamless Integration:** Designed to easily inject into standard transformer attention layers (`c_attn`, `q_proj`, `v_proj`) across popular architectures like GPT-2, Llama, and Mistral.

---

## 📦 Installation

Install QIA directly from your GitHub repository using `pip`:
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

# 5. Train Model
print("Starting QIA fine-tuning...")
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

print("\n--- Generated Output ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

```bash
pip install git+[https://github.com/Muhammadamin0334/Quantum-Inspired-Adaptation.git](https://github.com/Muhammadamin0334/Quantum-Inspired-Adaptation.git)
