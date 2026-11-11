
<div align="center">

# QIA: Quantum Inspired Adaptation

### A high-efficiency, lightweight fine-tuning framework for Large Language Models powered by reservoir computing and trigonometric wave mixing.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

</div>

---

## 📖 Overview

**QIA (Quantum Inspired Adaptation)** is an advanced, lightweight fine-tuning framework designed for Large Language Models (LLMs).

By freezing base model weights and leveraging structured phase-shift parameters alongside trigonometric wave mixing, QIA aims to stabilize output representations, suppress hallucinations, and achieve high training efficiency with minimal computational overhead.

The framework combines reservoir-inspired computation with phase-based parameterization to explore an alternative approach to parameter-efficient LLM adaptation.

---

## ✨ Key Features

- **Parameter-Efficient Training:** Freezes base model parameters and isolates updates to lightweight adapter layers and dual-component readout matrices.

- **Phase-Shift Parameterization:** Employs dynamic phase shifts and trigonometric wave mixing to amplify correct semantic trajectories and destructively interfere with noise.

- **Reservoir-Inspired Computation:** Integrates structured reservoir adapters into transformer attention layers.

- **Built-In Wave Regularization:** Includes a custom `QIRATrainer` featuring L2 regularization over phase-shift parameters to maintain mathematical stability during gradient descent.

- **Seamless Integration:** Designed to inject into standard transformer attention layers (`c_attn`, `q_proj`, `v_proj`) across supported architectures such as GPT-2, Llama, and Mistral.

---

## 📦 Installation

Install QIA directly from GitHub:

```bash
pip install git+https://github.com/Muhammadamin0334/Quantum-Inspired-Adaptation.git
```

### Dependencies

QIA requires:

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets 2.0+

---

## 🚀 Quick Start

The following example demonstrates how to integrate QIA into GPT-2 and fine-tune it using a small instruction dataset.

### 1. Import Libraries

```python
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from datasets import Dataset

# Import QIA framework
from qira import inject_qira, QIRATrainer
```

### 2. Load Base Model and Tokenizer

```python
MODEL_ID = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
```

### 3. Inject QIA Reservoir Adapters

QIA freezes the base model parameters and injects phase-mixing adapters into the selected attention layers.

```python
model = inject_qira(
    model,
    target_modules=["c_attn"],  # GPT-2
    reservoir_dim=128,
    alpha=0.03
)

# For Llama/Mistral, use:
# target_modules=["q_proj", "v_proj"]
```

### 4. Prepare Dataset

Create a simple instruction-response dataset.

```python
train_data = [
    {
        "instruction": "What is QIA?",
        "output": (
            "QIA is a quantum-inspired adaptation technique "
            "for efficient LLM fine-tuning and hallucination suppression."
        )
    },
    {
        "instruction": "How does QIA work?",
        "output": (
            "It utilizes phase-shift parameters and trigonometric "
            "wave mixing to amplify correct pathways and minimize output noise."
        )
    }
]

def format_data(example):

    prompt = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Response:\n{example['output']}"
    )

    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=128,
        padding="max_length"
    )

    labels = encoded["input_ids"].copy()

    # Ignore padding tokens when calculating the training loss
    labels = [
        token if mask == 1 else -100
        for token, mask in zip(labels, encoded["attention_mask"])
    ]

    encoded["labels"] = labels

    return encoded


train_dataset = Dataset.from_list(train_data).map(
    format_data,
    remove_columns=["instruction", "output"]
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8
)
```

### 5. Configure Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./qia_training_results",
    per_device_train_batch_size=2,
    num_train_epochs=5,
    learning_rate=3e-4,
    logging_steps=2,
    save_strategy="no",
    report_to="none"
)
```

### 6. Initialize QIA Trainer

The custom `QIRATrainer` incorporates phase-shift regularization into the training objective.

```python
trainer = QIRATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    phase_reg_weight=1e-3
)
```

### 7. Train the Model

```python
print("Starting QIA fine-tuning...")

trainer.train()
```

### 8. Test Generation

```python
model.eval()

test_prompt = (
    "### Instruction:\n"
    "What is QIA?\n\n"
    "### Response:\n"
)

inputs = tokenizer(
    test_prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

print("\n--- Generated Output ---")

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)
```

---

## 🧠 How QIA Works

QIA explores an alternative approach to conventional parameter-efficient fine-tuning through the combination of:

1. **Frozen Base Model:** Preserves pretrained model weights during adaptation.
2. **Reservoir Adapters:** Introduces lightweight trainable adapter components.
3. **Phase-Shift Parameters:** Provides a structured mechanism for modifying representations.
4. **Trigonometric Wave Mixing:** Uses phase-based transformations to combine learned signals.
5. **Phase Regularization:** Penalizes excessive phase-shift parameter values during optimization.

The overall objective combines the task-specific loss with a phase regularization term:

\[
\mathcal{L}_{QIA} =
\mathcal{L}_{task}+
\lambda \sum_{i}\theta_i^2
\]

Where:

- \(\mathcal{L}_{task}\): Original language-model training loss.
- \(\lambda\): Phase regularization coefficient.
- \(\theta_i\): Trainable phase-shift parameters.

---

## 📊 Evaluation

QIA is designed to investigate parameter-efficient LLM adaptation, particularly under resource-constrained and low-data conditions.

Important evaluation dimensions include:

- Trainable parameter count
- GPU memory consumption
- Training time
- Generation quality
- Repetition and token-collapse behaviour
- Hallucination-related evaluation
- Instruction-following performance

For comparative evaluation, QIA can be benchmarked against conventional approaches such as LoRA.

---

## 🔬 Research Motivation

Conventional parameter-efficient fine-tuning methods primarily focus on reducing the number of trainable parameters.

QIA explores a different direction by introducing phase-shift parameterization, reservoir-inspired computation, and trigonometric wave mixing into the adaptation process.

The objective is to investigate whether structured phase-based adaptation can improve generation stability and efficiency without requiring updates to the full pretrained model.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Muhammad Amin**

Researcher | Artificial Intelligence | Reservoir Computing | Quantum-Inspired Machine Learning

GitHub: [@Muhammadamin0334](https://github.com/Muhammadamin0334)

Project Repository:

[Quantum-Inspired-Adaptation](https://github.com/Muhammadamin0334/Quantum-Inspired-Adaptation)

---

⭐ If you find this project interesting, consider giving the repository a star!
