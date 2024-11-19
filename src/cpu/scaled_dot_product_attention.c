#include "nn.h"


__attribute__((weak)) void nn_scaled_dot_product_attention_f32(Tensor4D_F32 *y, const Tensor4D_F32 *query, const Tensor4D_F32 *key, const Tensor4D_F32 *value) {
  nn_assert(query->shape[0] == key->shape[0] && query->shape[0] == value->shape[0], "Query, key, and value must have the same batch size");
  nn_assert(query->shape[1] == key->shape[1] && query->shape[1] == value->shape[1], "Query, key, and value must have the same sequence length");
  nn_assert(query->shape[2] == key->shape[2] && query->shape[2] == value->shape[2], "Query, key, and value must have the same head count");
  nn_assert(query->shape[3] == key->shape[3] && query->shape[3] == value->shape[3], "Query, key, and value must have the same embedding dimension");

  size_t batch_size = query->shape[0];
  size_t sequence_length = query->shape[1];
  size_t head_count = query->shape[2];
  size_t embedding_dimension = query->shape[3];

  // L, S = query.size(-2), key.size(-2)
  size_t L = query->shape[2];
  size_t S = key->shape[2];

  // scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
  float scale_factor = 1 / sqrt(query->shape[3]);

  // attn_bias = torch.zeros(L, S, dtype=query.dtype)
  // if is_causal:
  //     assert attn_mask is None
  //     temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
  //     attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
  //     attn_bias.to(query.dtype)

  // if attn_mask is not None:
  //     if attn_mask.dtype == torch.bool:
  //         attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
  //     else:
  //         attn_bias += attn_mask
  
  // attn_weight = query @ key.transpose(-2, -1) * scale_factor
  nn_matmul4d_f32(y, query, nn_transpose4d_f32(key, 2, 3));
  // nn_scale4d_f32(y, scale_factor);
  
  // attn_weight += attn_bias
  // nn_add4d_f32(y, y, attn_bias);

  // attn_weight = torch.softmax(attn_weight, dim=-1)
  // nn_softmax4d_f32(y, y, -1);

  // attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
  // nn_dropout4d_f32(y, y, 0.5, 1);

  // return attn_weight @ value
  // nn_matmul4d_f32(y, y, value);
}
  