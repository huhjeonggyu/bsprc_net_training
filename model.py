# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 1000),  
            nn.ReLU(), 
            nn.Linear(1000, 1000),  
            nn.ReLU(), 
            nn.Linear(1000, 1) 
        )
    
    def forward(self, x):
        return self.layers(x)
