import torch

from einops import rearrange
from torch import nn
from lora_linear import LoRALinear


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.use_lora = config.use_lora
    if self.use_lora:
      r = config.lora_r
      alpha = config.lora_alpha
      lora_dropout = config.lora_dropout

      self.query = LoRALinear(config.hidden_size, self.all_head_size, bias=True, r=r, alpha=alpha, dropout=lora_dropout)
      self.key   = LoRALinear(config.hidden_size, self.all_head_size, bias=True, r=r, alpha=alpha, dropout=lora_dropout)
      self.value = LoRALinear(config.hidden_size, self.all_head_size, bias=True, r=r, alpha=alpha, dropout=lora_dropout)
    else:
      self.query = nn.Linear(config.hidden_size, self.all_head_size)
      self.key   = nn.Linear(config.hidden_size, self.all_head_size)
      self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    B, H, T, D = query.shape
    scale = D ** 0.5
    unmasked_scores = torch.matmul(query, key.transpose(-1, -2)) / scale
    
    #casual mask
    causal_mask = torch.triu(torch.ones(T, T, device=unmasked_scores.device, dtype=torch.bool), diagonal=1)
    scores = unmasked_scores.masked_fill(causal_mask, float('-inf'))
    
    scores = scores + attention_mask
    attn_probs = torch.softmax(scores, dim=-1)
    attn_probs = self.dropout(attn_probs)
    product = torch.matmul(attn_probs, value)
    output = rearrange(product, 'b h t d -> b t (h d)')
    return output

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
