import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModel

# Core Classes for Fine-Tuning
# These classes wrap the base transformer model and add custom layers or adapters.

class ClassificationModel(nn.Module):
    """
    A generic classification head to be placed on top of a transformer model.
    It takes the base model's last hidden state, extracts the CLS token,
    and passes it through a dropout and a linear layer for classification.
    """
    def __init__(self, base_model):
        super(ClassificationModel, self).__init__()
        
        self.base_model = base_model
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 2)  # 768 is the hidden size for DistilBERT, 2 is for positive/negative labels
        
    def forward(self, input_ids, attn_mask):
        last_hidden_state = self.base_model(input_ids, attention_mask=attn_mask).last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]  # Take the [CLS] token representation
        x = self.dropout(cls_embedding)
        logits = self.linear(x)
        return logits


class BottleneckAdapter(nn.Module):
    """
    A bottleneck adapter module that can be inserted into a transformer.
    
    It projects hidden states down to a lower-dimensional space and then 
    back up again, with non-linearity and dropout in between. This helps 
    the model adapt to new tasks without updating the original transformer.
    
    Args:
        hidden_size: The dimension of the model's hidden states (e.g., 768 for DistilBERT)
        adapter_size: The smaller bottleneck dimension (e.g., 64)
        dropout_rate: Regularization to improve generalization
    """
    def __init__(self, hidden_size, adapter_size, dropout_rate=0.1):
        super().__init__()
        
        self.down_project = nn.Linear(hidden_size, adapter_size)  # d -> b
        self.activation = nn.GELU()  # non-linearity
        self.up_project = nn.Linear(adapter_size, hidden_size)    # b -> d
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize adapter weights — not learned from pretraining, so good init is important!
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.down_project.bias)
        nn.init.xavier_uniform_(self.up_project.weight)
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
    """
    Wraps a DistilBERT TransformerBlock with adapters.
    """
    def __init__(self, transformer_layer, adapter_size):
        super().__init__()
        self.layer = transformer_layer
        self.hidden_size = transformer_layer.attention.q_lin.in_features

        # Freeze the original transformer block
        for param in self.layer.parameters():
            param.requires_grad = False

        # Add adapters
        self.attention_adapter = BottleneckAdapter(self.hidden_size, adapter_size)
        self.ffn_adapter = BottleneckAdapter(self.hidden_size, adapter_size)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # DistilBERT forward: attention -> add & norm -> ffn -> add & norm

        # 1. Attention sublayer
        sa_output = self.layer.attention(
            hidden_states, 
            attn_mask=attention_mask, 
            head_mask=head_mask
        )[0]

        # Add + Norm (frozen)
        sa_output = self.layer.sa_layer_norm(sa_output + hidden_states)

        # Adapter after attention
        sa_output = self.attention_adapter(sa_output)

        # 2. FFN sublayer
        ffn_output = self.layer.ffn(sa_output)
        ffn_output = self.layer.output_layer_norm(ffn_output + sa_output)

        # Adapter after FFN
        output = self.ffn_adapter(ffn_output)

        return output


class LoRALayer(nn.Module):
    """
    LoRA implementation for linear layers.
    """
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
    """
    Wraps a pre-trained linear layer with LoRA functionality.
    """
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


# Functions to Retrieve Different Fine-Tuning Models

def get_full_classification_model(base_model):
    """
    Adds a classification head to a base model for full fine-tuning.
    All parameters of the base model and the new head are trainable.
    """
    model = ClassificationModel(base_model)
    return model


def get_classification_head_model(base_model):
    """
    Adds a classification head and freezes the base model, so only
    the parameters of the new head are trainable.
    """
    for param in base_model.parameters():
        param.requires_grad = False
        
    model = ClassificationModel(base_model)
    return model


def get_adapters_model(base_model, adapter_size=64):
    """
    Replaces each layer in the base model with an AdapterTransformerLayer,
    which includes trainable bottleneck adapters. The base model's
    parameters are frozen.
    """
    # Create a new, modified base model with adapters
    for i in range(len(base_model.transformer.layer)):
        original_layer = base_model.transformer.layer[i]
        # Replace the original layer with the new adapter-wrapped layer
        base_model.transformer.layer[i] = AdapterTransformerLayer(original_layer, adapter_size)
    
    # Add the classification head to the modified base model
    classification_model = ClassificationModel(base_model)
    return classification_model


def get_lora_model(base_model, rank=8, alpha=32, target_modules=["q_lin", "v_lin"]):
    """
    Applies LoRA to specific modules in a transformer model,
    making only the LoRA weights trainable.
    
    Args:
        base_model: A Hugging Face transformer model
        rank: Rank for LoRA decomposition
        alpha: Scaling factor
        target_modules: List of module names to apply LoRA to.
    """
    # Freeze all parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Then apply LoRA to target modules
    for name, module in base_model.named_modules():
        if any(target_name in name for target_name in target_modules):
            if isinstance(module, nn.Linear):
                # Get the parent module and child name to set the attribute
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = base_model.get_submodule(parent_name)
                
                # Replace with LoRA version
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent_module, child_name, lora_layer)
    
    #  Add the classification head to the modified base model
    classification_model = ClassificationModel(base_model)
    return classification_model
