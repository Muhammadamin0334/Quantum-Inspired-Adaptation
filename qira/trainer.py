from transformers import Trainer
import torch

class QIRATrainer(Trainer):
    def __init__(self, *args, phase_reg_weight=1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_reg_weight = phase_reg_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        phase_reg = sum(torch.sum(p ** 2) for n, p in model.named_parameters() if "phase_shift" in n)
        total_loss = loss + (self.phase_reg_weight * phase_reg)
        return (total_loss, outputs) if return_outputs else total_loss
