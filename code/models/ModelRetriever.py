#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel

class ClassificationModel(nn.Module):
    def __init__(self, base_model):
        super(ClassificationModel, self).__init__()
        
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 2) # output features from BERT is 768 and 2 is number of labels
        
    def forward(self, input_ids, attn_mask):
        
        last_hidden_state = self.base_model(input_ids, attention_mask=attn_mask).last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]   # Take [CLS] token representation
        x = self.dropout(cls_embedding)
        logits = self.linear(x) 
        return logits

def get_full_classification_model(base_model):
    # Simply add classification head
    model = ClassificationModel(base_model)

    return model


def get_classification_head_model(base_model):
    # Freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False
            
    model = ClassificationModel(base_model)

    return model
    

class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_size, dropout_rate=0.1):
        super().__init__()
        
        self.down_project = nn.Linear(hidden_size, adapter_size)  # down projection
        self.activation = nn.ReLU()  # non-linearity
        self.up_project = nn.Linear(adapter_size, hidden_size)    # up projection
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize adapter weights — not learned from pretraining, so good init is important!
        nn.init.kaiming_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.kaiming_uniform_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, hidden_states):
        # Store original input for residual connection
        residual = hidden_states

        # Apply adapter: down-project -> non-linear -> up-project -> dropout
        x = self.down_project(hidden_states)
        x = self.activation(x)
        x = self.up_project(x)
        x = self.dropout(x)

        # Add residual and normalize
        output = residual + x
        output = self.layer_norm(output)
        return output

class AdapterTransformerLayer(nn.Module):
    def __init__(self, transformer_layer, adapter_size):
        super().__init__()
        self.layer = transformer_layer
        self.hidden_size = transformer_layer.attention.self.query.in_features

        # Freeze the original transformer block
        for param in self.layer.parameters():
            param.requires_grad = False

        # Add adapters
        self.attention_adapter = BottleneckAdapter(self.hidden_size, adapter_size)
        self.ffn_adapter = BottleneckAdapter(self.hidden_size, adapter_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # BERT forward: attention -> add & norm -> ffn -> add & norm

        # Attention sublayer
        sa_output = self.layer.attention(
            hidden_states, 
            attn_mask=attention_mask, 
            head_mask=head_mask
        )[0]

        # Add + Norm (frozen)
        sa_output = self.layer.sa_layer_norm(sa_output + hidden_states)

        # Adapter after attention
        sa_output = self.attention_adapter(sa_output)

        # FFN sublayer
        ffn_output = self.layer.ffn(sa_output)
        ffn_output = self.layer.output_layer_norm(ffn_output + sa_output)

        # Adapter after FFN
        output = self.ffn_adapter(ffn_output)

        return output

def get_adapters_model(base_model, adapter_size=64):
    for i in range(len(base_model.encoder.layer)):
        original_layer = base_model.encoder.layer[i]
        base_model.encoder.layer[i] = AdapterTransformerLayer(original_layer, adapter_size)
    
    classification_model = ClassificationModel(base_model)
    return classification_model



class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        # LoRA weights
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # LoRA contribution: scaling * (x @ A) @ B
        return self.scaling * (x @ self.lora_A) @ self.lora_B


class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank=8, alpha=32):
        super().__init__()
        self.linear = linear_layer
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # Add LoRA components
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features,
            rank=rank,
            alpha=alpha
        )
    
    def forward(self, x):
        # Combine original output with LoRA contribution
        return self.linear(x) + self.lora(x)


def get_lora_model(base_model, rank=8, alpha=32, target_modules=["query", "value"]):
    # First, freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Then apply LoRA to target modules
    for name, module in base_model.named_modules():
        if any(target_name in name for target_name in target_modules):
            if isinstance(module, nn.Linear):
                # Get the parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = base_model.get_submodule(parent_name)
                
                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent_module, child_name, lora_layer)
    
    classification_model = ClassificationModel(base_model)
    return classification_model





