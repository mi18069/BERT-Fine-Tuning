#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel
from copy import deepcopy

class ClassificationModel(nn.Module):
    def __init__(self, base_model):
        super(ClassificationModel, self).__init__()
        
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 2) # output features from BERT is 768 and 2 is number of labels
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
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
    

class AdapterModule(nn.Module):
    def __init__(self, in_feature, adapter_size=64):
        super().__init__()

        self.proj_down = nn.Linear(in_features=in_feature, out_features=adapter_size)
        self.proj_up = nn.Linear(in_features=adapter_size, out_features=in_feature)
        
        # Setting initial values to 0 in order not to interfere with models existing knowledge
        nn.init.zeros_(self.proj_up.weight)
        nn.init.zeros_(self.proj_up.bias)

    def forward(self, x):
        return self.proj_up(F.relu(self.proj_down(x))) + x

class BertLayerWithAdapters(nn.Module):
    def __init__(self, base_layer, adapter_size=64):
        super().__init__()

        self.base_layer = deepcopy(base_layer)
        self.adapter_size = adapter_size
        hidden_size = self.base_layer.output.dense.out_features
        self.attention_adapter = AdapterModule(hidden_size, adapter_size)
        self.ffn_adapter = AdapterModule(hidden_size, adapter_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        cache_position=None,
    ):

        # Call the base layer's attention sub-module
        sa_output = self.base_layer.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )[0]
        
        # Apply the attention adapter and add its output to the residual path
        adapter_attention_output = self.attention_adapter(sa_output)
        attention_output = hidden_states + sa_output + adapter_attention_output
        attention_output = self.base_layer.output.LayerNorm(attention_output)

        # Call the base layer's feedforward sub-modules
        intermediate_output = self.base_layer.intermediate(attention_output)
        ffn_output = self.base_layer.output.dense(intermediate_output)
        
        # Apply the FFN adapter and add its output to the residual path
        adapter_ffn_output = self.ffn_adapter(ffn_output)
        layer_output = attention_output + ffn_output + adapter_ffn_output
        layer_output = self.base_layer.output.LayerNorm(layer_output)
        
        # The return value should match the original BERT layer's output format
        return (layer_output, ) + self.base_layer.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )[1:]

def get_adapters_model(base_model, adapter_size=64):
    for i, layer in enumerate(base_model.encoder.layer):
        base_model.encoder.layer[i] = BertLayerWithAdapters(layer, adapter_size=64)
        
    for name, param in base_model.named_parameters():
        param.requires_grad = False
    for name, param in base_model.named_parameters():
        if "attention_adapter" in name or "ffn_adapter" in name:
            param.requires_grad = True
            
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





