
import math

import torch

# seed
torch.manual_seed(0)


batch_size = 1
dim = 4
max_seq_len = 2

q = torch.randn(batch_size, max_seq_len, dim)
k = torch.randn(batch_size, max_seq_len, dim)
v = torch.randn(batch_size, max_seq_len, dim)


# class Attention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         n_heads = 1
#         dim = 8
#         max_seq_len = 4

#         self.n_kv_heads = 1
#         model_parallel_size = 1
#         self.n_local_heads = n_heads // model_parallel_size
#         self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
#         self.n_rep = self.n_local_heads // self.n_local_kv_heads
#         self.head_dim = dim // n_heads
#         self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
#         self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
#         self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
#         self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
        
#         print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
#         mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
#         mask = torch.triu(mask, diagonal=1)
#         self.register_buffer("mask", mask)

#     def forward(
#         self,
#         x: torch.Tensor,
#         freqs_cos: torch.Tensor,
#         freqs_sin: torch.Tensor,
#     ):
#         bsz, seqlen, _ = x.shape

#         # QKV
#         xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
#         xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
#         xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

#         # RoPE relative positional embeddings
#         xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

#         # grouped multiquery attention: expand out keys and values
#         xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
#         xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

#         # make heads into a batch dimension
#         xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
#         xk = xk.transpose(1, 2)
#         xv = xv.transpose(1, 2)

#         # flash implementation
#         if self.flash:
#             output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
#         else:
#             # manual implementation
#             scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
#             assert hasattr(self, 'mask')
#             scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
#             scores = F.softmax(scores.float(), dim=-1).type_as(xq)
#             scores = self.attn_dropout(scores)
#             output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

#         # restore time as batch dimension and concat heads
#         output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

#         # final projection into the residual stream
#         output = self.wo(output)
#         output = self.resid_dropout(output)
#         return output





def DotProductAttention(query, key, value, scale=0):
    # q, k, v shape: (batch_size, seq_len, head_dim)
    
    l = query.shape[-2]
    s = key.shape[-2]

    d_k = query.shape[-1]


    scale_factor = 1 / math.sqrt(d_k) if scale == 0 else scale
    attn_bias = torch.zeros(l, s, dtype=query.dtype)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    result = attn_weight @ value

    return result


# # Efficient implementation equivalent to the following:
# def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#     attn_bias = torch.zeros(L, S, dtype=query.dtype)
#     if is_causal:
#         assert attn_mask is None
#         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#         attn_bias.to(query.dtype)

#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#         else:
#             attn_bias += attn_mask
#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value


result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)

result2 = DotProductAttention(q, k, v, scale=0)



print(result)
print(result2)