import torch
import torch.nn as nn
import math

class QuantumInspiredReservoirAdapter(nn.Module):
    def __init__(self, in_features, out_features, reservoir_dim=256, alpha=0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.reservoir_dim = reservoir_dim
        self.alpha = alpha
        
        self.register_buffer("W_in", torch.randn(in_features, reservoir_dim) / math.sqrt(in_features))
        self.register_buffer("W_res", torch.randn(reservoir_dim, reservoir_dim) / math.sqrt(reservoir_dim))
        
        self.phase_shift = nn.Parameter(torch.zeros(reservoir_dim))
        self.readout_real = nn.Parameter(torch.randn(reservoir_dim, out_features) / math.sqrt(reservoir_dim))
        self.readout_imag = nn.Parameter(torch.randn(reservoir_dim, out_features) / math.sqrt(reservoir_dim))

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)
        
        state_in = torch.matmul(x_flat, self.W_in)
        state_res = torch.tanh(torch.matmul(state_in, self.W_res) + state_in)
        
        phases = state_res + self.phase_shift
        psi_real = torch.cos(phases) * state_res
        psi_imag = torch.sin(phases) * state_res
        
        out_real = torch.matmul(psi_real, self.readout_real) - torch.matmul(psi_imag, self.readout_imag)
        out_imag = torch.matmul(psi_real, self.readout_imag) + torch.matmul(psi_imag, self.readout_real)
        interference_out = out_real + out_imag
        
        return (self.alpha * interference_out).reshape(*batch_shape, self.out_features)

class QIRALinearWrapper(nn.Module):
    def __init__(self, base_layer, reservoir_dim=256, alpha=0.05):
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        self.qira_adapter = QuantumInspiredReservoirAdapter(
            base_layer.in_features, base_layer.out_features, reservoir_dim, alpha
        )

    def forward(self, x):
        return self.base_layer(x) + self.qira_adapter(x)

def inject_qira(model, target_modules=["c_attn"], reservoir_dim=256, alpha=0.05):
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition('.')
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, QIRALinearWrapper(module, reservoir_dim, alpha))
    return model
