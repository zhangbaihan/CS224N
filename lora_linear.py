
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """
    Replacement for nn.Linear with LoRA.
    Keeps a frozen base W,b and learns low-rank A,B.
    """
    def __init__(self, in_features, out_features, bias=True, r=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Linear
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA params
        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.A, a=(5 ** 0.5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        y = self.base(x)
        lora = (self.dropout(x) @ self.A.t()) @ self.B.t()
        return y + self.scaling * lora

    @torch.no_grad()
    def merge(self):
        self.base.weight += self.scaling * (self.B @ self.A)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias