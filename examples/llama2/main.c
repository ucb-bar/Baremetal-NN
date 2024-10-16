/* Inference for Llama-2 Transformer model in pure C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

#include <unistd.h>
// ----------------------------------------------------------------------------
// Transformer model

#include "nn.h"
#include "riscv.h"


// load the weight data block from the model.bin file
INCLUDE_FILE(".rodata", "./checkpoints/tokenizer.bin", tokenizer);
extern uint8_t tokenizer_data[];
extern size_t tokenizer_start[];
extern size_t tokenizer_end[];

INCLUDE_FILE(".rodata", "./checkpoints/stories15M.bin", checkpoint);
extern uint8_t checkpoint_data[];
extern size_t checkpoint_start[];
extern size_t checkpoint_end[];


typedef struct {
  int dim; // transformer dimension
  int hidden_dim; // for ffn layers
  int n_layers; // number of layers
  int n_heads; // number of query heads
  int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
  int vocab_size; // vocabulary size, usually 256 (byte-level)
  int seq_len; // max sequence length
} Config;

typedef struct {
  // token embedding table
  float *token_embedding_table_ptr;    // (vocab_size, dim)
  // weights for rmsnorms
  float *rms_att_weight_ptr; // (layer, dim) rmsnorm weights
  float *rms_ffn_weight_ptr; // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  float *wq_ptr; // (layer, dim, n_heads * head_size)
  float *wk_ptr; // (layer, dim, n_kv_heads * head_size)
  float *wv_ptr; // (layer, dim, n_kv_heads * head_size)
  float *wo_ptr; // (layer, n_heads * head_size, dim)
  // weights for ffn
  float *w1_ptr; // (layer, hidden_dim, dim)
  float *w2_ptr; // (layer, dim, hidden_dim)
  float *w3_ptr; // (layer, hidden_dim, dim)
  // final rmsnorm
  float *rms_final_weight_ptr; // (dim,)
  // (optional) classifier weights for the logits, on the last layer
  float *wcls_ptr;
} TransformerWeights;

typedef struct {
  Tensor1D_F32 *rms_final_weight;
  Tensor2D_F32 *wq;
  Tensor2D_F32 *wk;
  Tensor2D_F32 *wv;
  Tensor2D_F32 *wo;
  Tensor2D_F32 *w1;
  Tensor2D_F32 *w2;
  Tensor2D_F32 *w3;
  Tensor2D_F32 *wcls;

  // current wave of activations
  Tensor1D_F32 *x;     // activation at current time stamp (dim,)
  Tensor1D_F32 *xb;    // same, but inside a residual branch (dim,)
  Tensor1D_F32 *xb2;   // an additional buffer just for convenience (dim,)
  Tensor1D_F32 *hb;    // buffer for hidden dimension in the ffn (hidden_dim,)
  Tensor1D_F32 *hb2;   // buffer for hidden dimension in the ffn (hidden_dim,)
  Tensor1D_F32 *q;     // query (dim,)
  Tensor1D_F32 *k;     // key (dim,)
  Tensor1D_F32 *v;     // value (dim,)
  Tensor2D_F32 *att; // buffer for scores/attention values (n_heads, seq_len)
  Tensor1D_F32 *logits; // output logits
  // kv cache
  Tensor3D_F32 *key_cache;   // (layer, seq_len, dim)
  Tensor3D_F32 *value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
  Config config; // the hyperparameters of the architecture (the blueprint)
  TransformerWeights weights; // the weights of the model
  RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;


void malloc_run_state(RunState *s, Config *p) {
  // we calloc instead of malloc to keep valgrind happy
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  s->key_cache = nn_zeros(3, (size_t[]){p->n_layers, p->seq_len, kv_dim}, DTYPE_F32);
  s->value_cache = nn_zeros(3, (size_t[]){p->n_layers, p->seq_len, kv_dim}, DTYPE_F32);
  s->att = nn_zeros(2, (size_t[]){p->n_heads, p->seq_len}, DTYPE_F32);
  s->logits = nn_zeros(1, (size_t[]){p->vocab_size}, DTYPE_F32);

  s->x = nn_zeros(1, (size_t[]){p->dim}, DTYPE_F32);
  s->xb = nn_zeros(1, (size_t[]){p->dim}, DTYPE_F32);
  s->xb2 = nn_zeros(1, (size_t[]){p->dim}, DTYPE_F32);
  s->hb = nn_zeros(1, (size_t[]){p->hidden_dim}, DTYPE_F32);
  s->hb2 = nn_zeros(1, (size_t[]){p->hidden_dim}, DTYPE_F32);

  s->q = nn_tensor(1, (size_t[]){p->dim}, DTYPE_F32, NULL);
  s->k = nn_tensor(1, (size_t[]){kv_dim}, DTYPE_F32, NULL);
  s->v = nn_tensor(1, (size_t[]){kv_dim}, DTYPE_F32, NULL);
}

void free_run_state(RunState *s) {
  // free(s->x);
  // free(s->xb);
  // free(s->xb2);
  // free(s->hb);
  // free(s->hb2);
  //   free(s->q);
  //   free(s->att);
  //   free(s->logits);
  //   free(s->key_cache);
  //   free(s->value_cache);
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
  int head_size = p->dim / p->n_heads;
  // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
  unsigned long long n_layers = p->n_layers;
  w->token_embedding_table_ptr = ptr;
  ptr += p->vocab_size * p->dim;
  w->rms_att_weight_ptr = ptr;
  ptr += n_layers * p->dim;
  w->wq_ptr = ptr;
  ptr += n_layers * p->dim * (p->n_heads * head_size);
  w->wk_ptr = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wv_ptr = ptr;
  ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
  w->wo_ptr = ptr;
  ptr += n_layers * (p->n_heads * head_size) * p->dim;
  w->rms_ffn_weight_ptr = ptr;
  ptr += n_layers * p->dim;
  w->w1_ptr = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->w2_ptr = ptr;
  ptr += n_layers * p->hidden_dim * p->dim;
  w->w3_ptr = ptr;
  ptr += n_layers * p->dim * p->hidden_dim;
  w->rms_final_weight_ptr = ptr;
  ptr += p->dim;
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
  ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
  w->wcls_ptr = shared_weights ? w->token_embedding_table_ptr : ptr;
}

void read_checkpoint(Transformer *t) {
  Config *config = &t->config;
  TransformerWeights *weights = &t->weights;

  memcpy(config, checkpoint_data, sizeof(Config));

  // negative vocab size is hacky way of signaling unshared weights. bit yikes.
  int shared_weights = config->vocab_size > 0 ? 1 : 0;
  config->vocab_size = abs(config->vocab_size);
  size_t file_size = (size_t)checkpoint_end - (size_t)checkpoint_start;

  float *weights_ptr = (float *)checkpoint_data + sizeof(Config)/sizeof(float);
  memory_map_weights(weights, config, weights_ptr, shared_weights);
}

// void init(Model *model) {
void init_transformer(Transformer *t) {
  read_checkpoint(t);
  malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
  free_run_state(&t->state);
}



Tensor *forward(Transformer *transformer, int token, int pos) {
  // a few convenience variables
  Config *p = &transformer->config;
  TransformerWeights *w = &transformer->weights;
  RunState *s = &transformer->state;

  int dim = p->dim;
  int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
  int hidden_dim =  p->hidden_dim;
  int head_size = dim / p->n_heads;

  // copy the token embedding into x
  float *content_row = w->token_embedding_table_ptr + token * dim;
  memcpy(s->x->data, content_row, dim*sizeof(float));

  s->rms_final_weight = nn_tensor(1, (size_t[]){dim}, DTYPE_F32, w->rms_ffn_weight_ptr);
  s->rms_final_weight = nn_tensor(1, (size_t[]){dim}, DTYPE_F32, w->rms_att_weight_ptr);
  s->wq = nn_tensor(2, (size_t[]){dim, dim}, DTYPE_F32, w->wq_ptr);
  s->wk = nn_tensor(2, (size_t[]){kv_dim, dim}, DTYPE_F32, w->wk_ptr);
  s->wv = nn_tensor(2, (size_t[]){kv_dim, dim}, DTYPE_F32, w->wv_ptr);
  s->wo = nn_tensor(2, (size_t[]){dim, dim}, DTYPE_F32, w->wo_ptr);
  s->w1 = nn_tensor(2, (size_t[]){hidden_dim, dim}, DTYPE_F32, w->w1_ptr);
  s->w2 = nn_tensor(2, (size_t[]){dim, hidden_dim}, DTYPE_F32, w->w2_ptr);
  s->w3 = nn_tensor(2, (size_t[]){hidden_dim, dim}, DTYPE_F32, w->w3_ptr);

  s->rms_final_weight = nn_tensor(1, (size_t[]){p->dim}, DTYPE_F32, w->rms_final_weight_ptr);
  s->wcls = nn_tensor(2, (size_t[]){p->vocab_size, p->dim}, DTYPE_F32, w->wcls_ptr);

  Tensor *att_tensor = nn_tensor(2, (size_t[]){1, pos + 1}, DTYPE_F32, s->att->data);

  // forward all the layers
  for (size_t l = 0; l < p->n_layers; l += 1) {
    // attention rmsnorm
    s->rms_final_weight->data = w->rms_att_weight_ptr + l*dim;
    nn_rms_norm(s->xb, s->x, s->rms_final_weight, 1e-5);

    // key and value point to the kv cache
    int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
    s->k->data = (float *)s->key_cache->data + loff + pos * kv_dim;
    s->v->data = (float *)s->value_cache->data + loff + pos * kv_dim;

    // qkv matmuls for this position
    s->wq->data = w->wq_ptr + l*dim*dim;
    s->wk->data = w->wk_ptr + l*dim*kv_dim;
    s->wv->data = w->wv_ptr + l*dim*kv_dim;
    
    nn_matmul(s->q, s->wq, s->xb);
    nn_matmul(s->k, s->wk, s->xb);
    nn_matmul(s->v, s->wv, s->xb);
  

    // RoPE relative positional encoding: complex-valued rotate q and k in each head
    for (size_t i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
      for (size_t v = 0; v < rotn; v += 1) {
        float *vec = v == 0 ? (float *)s->q->data : (float *)s->k->data; // the vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
      }
    }

    // multihead attention. iterate over all heads
    for (size_t h = 0; h < p->n_heads; h += 1) {
      // get the query vector for this head
      float *q = ((float *)s->q->data) + h * head_size;
      // attention scores for this head
      float *att = (float *)s->att->data + h * p->seq_len;
      // iterate over all timesteps, including the current one
      for (size_t t = 0; t <= pos; t += 1) {
        // get the key vector for this head and at this timestep
        float *k = (float *)s->key_cache->data + loff + t * kv_dim + (h / kv_mul) * head_size;
        // calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (size_t i = 0; i < head_size; i += 1) {
          score += q[i] * k[i];
        }
        score /= sqrtf(head_size);
        // save the score to the attention buffer
        att[t] = score;
      }

      // softmax the scores to get attention weights, from 0..pos inclusively
      att_tensor->data = att;
      nn_softmax(att_tensor, att_tensor, 1);

      // weighted sum of the values, store back into xb
      float *xb = ((float *)s->xb->data) + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (size_t t = 0; t <= pos; t += 1) {
        // get the value vector for this head and at this timestep
        float *v = (float *)s->value_cache->data + loff + t * kv_dim + (h / kv_mul) * head_size;
        // get the attention weight for this timestep
        float a = att[t];
        // accumulate the weighted value into xb
        for (size_t i = 0; i < head_size; i += 1) {
          xb[i] += a * v[i];
        }
      }
    }

    // final matmul to get the output of the attention
    s->wo->data = w->wo_ptr + l*dim*dim;
    nn_matmul(s->xb2, s->wo, s->xb);

    // residual connection back into x
    nn_add_inplace(s->x, s->xb2);

    // ffn rmsnorm
    s->rms_final_weight->data = w->rms_ffn_weight_ptr + l*dim;
    nn_rms_norm(s->xb, s->x, s->rms_final_weight, 1e-5);

    // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
    // first calculate self.w1(x) and self.w3(x)
    s->w1->data = w->w1_ptr + l*dim*hidden_dim;
    s->w3->data = w->w3_ptr + l*dim*hidden_dim;
    nn_matmul(s->hb, s->w1, s->xb);
    nn_matmul(s->hb2, s->w3, s->xb);

    // SwiGLU non-linearity
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    nn_silu(s->hb, s->hb);
    // elementwise multiply with w3(x)
    nn_mul_inplace(s->hb, s->hb2);
    
    // final matmul to get the output of the ffn
    s->w2->data = w->w2_ptr + l*dim*hidden_dim;
    nn_matmul(s->xb, s->w2, s->hb);

    // residual connection
    nn_add_inplace(s->x, s->xb);
  }

  // final rmsnorm
  nn_rms_norm(s->x, s->x, s->rms_final_weight, 1e-5);

  // classifier into logits
  nn_matmul(s->logits, s->wcls, s->x);

  return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
  char *str;
  int id;
} TokenIndex;

typedef struct {
  char **vocab;
  float *vocab_scores;
  TokenIndex *sorted_vocab;
  int vocab_size;
  unsigned int max_token_length;
  unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
  return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void init_tokenizer(Tokenizer *t, int vocab_size) {
// i should have written the vocab_size into the tokenizer file... sigh
t->vocab_size = vocab_size;
// malloc space to hold the scores and the strings
t->vocab = (char**)malloc(vocab_size * sizeof(char*));
t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
t->sorted_vocab = NULL; // initialized lazily
for (int i = 0; i < 256; i++) {
  t->byte_pieces[i * 2] = (unsigned char)i;
  t->byte_pieces[i * 2 + 1] = '\0';
}

uint8_t *tokenizer_ptr = tokenizer_data;

t->max_token_length = *(int*)tokenizer_ptr;
tokenizer_ptr += sizeof(int);

int len;
for (int i = 0; i < vocab_size; i++) {
  t->vocab_scores[i] = *(float*)(tokenizer_ptr);
  tokenizer_ptr += sizeof(float);

  len = *(int*)(tokenizer_ptr);
  tokenizer_ptr += sizeof(int);

  t->vocab[i] = (char *)malloc(len + 1);
  
  memcpy(t->vocab[i], tokenizer_ptr, len);
  tokenizer_ptr += len;

  t->vocab[i][len] = '\0'; // add the string terminating token
}
}

void free_tokenizer(Tokenizer *t) {
for (size_t i = 0; i < t->vocab_size; i += 1) {
  free(t->vocab[i]);
}
free(t->vocab);
free(t->vocab_scores);
free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
  char *piece = t->vocab[token];
  // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
  if (prev_token == 1 && piece[0] == ' ') {
    piece += 1;
  }
  // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
  // parse this and convert and return the actual byte
  unsigned char byte_val;
  if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
    piece = (char *)t->byte_pieces + byte_val * 2;
  }
  return piece;
}

void safe_printf(char *piece) {
// piece might be a raw byte token, and we only want to print printable chars or whitespace
// because some of the other bytes can be various control codes, backspace, etc.
if (piece == NULL) { return; }
if (piece[0] == '\0') { return; }
if (piece[1] == '\0') {
    unsigned char byte_val = piece[0];
    if (!(isprint(byte_val) || isspace(byte_val))) {
        return; // bad byte, don't print it
    }
}
printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
// efficiently find the perfect match for str in vocab, return its index or -1 if not found
TokenIndex tok = { .str = str }; // acts as the key to search for
TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
  // encode the string text (input) into an upper-bound preallocated tokens[] array
  // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
  if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

  if (t->sorted_vocab == NULL) {
      // lazily malloc and sort the vocabulary
      t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
      for (int i = 0; i < t->vocab_size; i++) {
          t->sorted_vocab[i].str = t->vocab[i];
          t->sorted_vocab[i].id = i;
      }
      qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
  }

  // create a temporary buffer that will store merge candidates of always two consecutive tokens
  // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
  char *str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
  size_t str_len = 0;

  // start at 0 tokens
  *n_tokens = 0;

  // add optional BOS (=1) token, if desired
  if (bos) tokens[(*n_tokens)++] = 1;

  // add_dummy_prefix is true by default
  // so prepend a dummy prefix token to the input string, but only if text != ""
  // TODO: pretty sure this isn't correct in the general case but I don't have the
  // energy to read more of the sentencepiece code to figure out what it's doing
  if (text[0] != '\0') {
    int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
    tokens[(*n_tokens)++] = dummy_prefix;
  }

  // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
  // Code point ↔ UTF-8 conversion
  // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
  // U+0000	U+007F	    0xxxxxxx
  // U+0080	U+07FF	    110xxxxx	10xxxxxx
  // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
  // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

  // process the raw (UTF-8) byte sequence of the input string
  for (char *c = text; *c != '\0'; c++) {

    // reset buffer if the current byte is ASCII or a leading byte
    // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
    // 0x80 is 10000000
    // in UTF-8, all continuation bytes start with "10" in first two bits
    // so in English this is: "if this byte is not a continuation byte"
    if ((*c & 0xC0) != 0x80) {
        // this byte must be either a leading byte (11...) or an ASCII char (0x...)
        // => reset our location, as we're starting a new UTF-8 codepoint
        str_len = 0;
    }

    // append the current byte to the buffer
    str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
    str_buffer[str_len] = '\0';

    // while the next character is a continuation byte, continue appending
    // but if there are too many of them, just stop to avoid overruning str_buffer size.
    if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
      continue;
    }

    // ok c+1 is not a continuation byte, so we've read in a full codepoint
    int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

    if (id != -1) {
      // we found this codepoint in vocab, add it as a token
      tokens[(*n_tokens)++] = id;
    }
    else {
      // byte_fallback encoding: just encode each byte as a token
      // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
      // so the individual bytes only start at index 3
      for (int i=0; i < str_len; i++) {
        tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
      }
    }
    str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
  }

  // merge the best consecutive pair each iteration, according the scores in vocab_scores
  while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (size_t i=0; i < (*n_tokens-1); i += 1) {
      // check if we can merge the pair (tokens[i], tokens[i+1])
      sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
      int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
      if (id != -1 && t->vocab_scores[id] > best_score) {
        // this merge pair exists in vocab! record its score and position
        best_score = t->vocab_scores[id];
        best_id = id;
        best_idx = i;
      }
    }

    if (best_idx == -1) {
      break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx+1; i < (*n_tokens-1); i++) {
        tokens[i] = tokens[i+1];
    }
    *n_tokens -= 1; // token length decreased
  }

  // add optional EOS (=2) token, if desired
  if (eos) tokens[(*n_tokens)++] = 2;

  free(str_buffer);
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
  float prob;
  int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
  int vocab_size;
  ProbIndex *probindex; // buffer used in top-p sampling
  float temperature;
  float topp;
  unsigned long long rng_state;
} Sampler;

size_t sample_argmax(Tensor *probabilities) {
  // return the index that has the highest probability
  size_t n = probabilities->shape[0];
  size_t max_i = 0;
  float max_p = ((float *)probabilities->data)[0];
  for (size_t i = 1; i < n; i += 1) {
    if (((float *)probabilities)[i] > max_p) {
      max_i = i;
      max_p = ((float *)probabilities)[i];
    }
  }
  return max_i;
}

size_t sample_mult(Tensor *probabilities, float coin) {
  // sample index from probabilities (they must sum to 1!)
  // coin is a random number in [0, 1), usually from random_f32()
  size_t n = probabilities->shape[0];
  float cdf = 0.0f;
  for (size_t i = 0; i < n; i += 1) {
    cdf += ((float *)probabilities->data)[i];
    if (coin < cdf) {
      return i;
    }
  }
  return n - 1; // in case of rounding errors
}

int compare(const void *a, const void *b) {
  ProbIndex *a_ = (ProbIndex *) a;
  ProbIndex *b_ = (ProbIndex *) b;
  if (a_->prob > b_->prob) return -1;
  if (a_->prob < b_->prob) return 1;
  return 0;
}

size_t sample_topp(Tensor *probabilities, float topp, ProbIndex *probindex, float coin) {
  // top-p sampling (or "nucleus sampling") samples from the smallest set of
  // tokens that exceed probability topp. This way we never sample tokens that
  // have very low probabilities and are less likely to go "off the rails".
  // coin is a random number in [0, 1), usually from random_f32()
  size_t n = probabilities->shape[0];
  size_t n0 = 0;
  // quicksort indices in descending order of probabilities
  // values smaller than (1 - topp) / (n - 1) cannot be part of the result
  // so for efficiency we crop these out as candidates before sorting
  const float cutoff = (1.0f - topp) / (n - 1);
  for (size_t i = 0; i < n; i += 1) {
    if (((float *)probabilities->data)[i] >= cutoff) {
      probindex[n0].index = i;
      probindex[n0].prob = ((float *)probabilities->data)[i];
      n0 += 1;
    }
  }
  qsort(probindex, n0, sizeof(ProbIndex), compare);

  // truncate the list where cumulative probability exceeds topp
  float cumulative_prob = 0.0f;
  size_t last_idx = n0 - 1; // in case of rounding errors consider all elements
  for (size_t i = 0; i < n0; i += 1) {
    cumulative_prob += probindex[i].prob;
    if (cumulative_prob > topp) {
      last_idx = i;
      break; // we've exceeded topp by including last_idx
    }
  }

  // sample from the truncated list
  float r = coin * cumulative_prob;
  float cdf = 0.0f;
  for (size_t i = 0; i <= last_idx; i += 1) {
    cdf += probindex[i].prob;
    if (r < cdf) {
      return probindex[i].index;
    }
  }
  return probindex[last_idx].index; // in case of rounding errors
}

void init_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
sampler->vocab_size = vocab_size;
sampler->temperature = temperature;
sampler->topp = topp;
sampler->rng_state = rng_seed;
// buffer only used with nucleus sampling; may not need but it's ~small
sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
// xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
*state ^= *state >> 12;
*state ^= *state << 25;
*state ^= *state >> 27;
return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
  return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, Tensor *logits) {
  // sample the token given the logits and some hyperparameters
  int next;
  if (sampler->temperature == 0.0f) {
    // greedy argmax sampling: take the token with the highest probability
    next = sample_argmax(logits);
  }
  else {
    // apply the temperature to the logits
    nn_mul1_inplace(logits, 1.0f / sampler->temperature);

    // apply softmax to the logits to get the probabilities for next token
    nn_softmax(logits, logits, 1);
    
    // flip a (float) coin (this is our source of entropy for sampling)
    float coin = random_f32(&sampler->rng_state);
    // we sample from this distribution to get the next token
    if (sampler->topp <= 0 || sampler->topp >= 1) {
      // simply sample from the predicted probability distribution
      next = sample_mult(logits, coin);
    } else {
      // top-p (nucleus) sampling, clamping the least likely tokens to zero
      next = sample_topp(logits, sampler->topp, sampler->probindex, coin);
    }
  }
  return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
  // return time in milliseconds, for benchmarking the model speed
  struct timespec time;
  // clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
  char *empty_prompt = "";
  if (prompt == NULL) {
    prompt = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  long start = 0;  // used to time our code, only initialized after first iteration
  int next;        // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;     // position in the sequence

  while (pos < steps) {
    // forward the transformer to get logits for the next token
    size_t cycles = READ_CSR("mcycle");
    Tensor *logits = forward(transformer, token, pos);
    cycles = READ_CSR("mcycle") - cycles;
    printf("forward taking %ld cycles\n", cycles);

    // advance the state machine
    if (pos < num_prompt_tokens - 1) {
      // if we are still processing the input prompt, force the next prompt token
      next = prompt_tokens[pos + 1];
    } else {
      // otherwise sample the next token from the logits
      next = sample(sampler, logits);
    }
    pos += 1;

    // data-dependent terminating condition: the BOS (=1) token delimits sequences
    if (next == 1) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    char *piece = decode(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    // fflush(stdout);
    printf("\n");
    token = next;

    // init the timer here because the first iteration can be slower
    if (start == 0) {
      start = time_in_ms();
    }
  }
  printf("\n");

  // report achieved tok/s (pos-1 because the timer starts after first iteration)
  if (pos > 1) {
    long end = time_in_ms();
    fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
  }

  free(prompt_tokens);
}

void read_stdin(const char *guide, char *buffer, size_t bufsize) {
  // read a line from stdin, up to but not including \n
  printf("%s", guide);
  if (fgets(buffer, bufsize, stdin) != NULL) {
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0'; // strip newline
    }
  }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
        char *cli_user_prompt, char *cli_system_prompt, int steps) {

  // buffers for reading the system prompt and user prompt from stdin
  // you'll notice they are soomewhat haphazardly and unsafely set atm
  char system_prompt[512];
  char user_prompt[512];
  char rendered_prompt[1152];
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int*)malloc(1152 * sizeof(int));
  int user_idx;

  // start the main loop
  int8_t user_turn = 1; // user starts
  int next = 0;        // will store the next token in the sequence
  int token = 0;       // stores the current token to feed into the transformer
  int prev_token = 0;
  int pos = 0;     // position in the sequence
  while (pos < steps) {

      // when it is the user's turn to contribute tokens to the dialog...
      if (user_turn) {
          // get the (optional) system prompt at position 0
          if (pos == 0) {
              // at position 0, the user can also contribute a system prompt
              if (cli_system_prompt == NULL) {
                  // system prompt was not passed in, attempt to get it from stdin
                  read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
              } else {
                  // system prompt was passed in, use it
                  strcpy(system_prompt, cli_system_prompt);
              }
          }
          // get the user prompt
          if (pos == 0 && cli_user_prompt != NULL) {
              // user prompt for position 0 was passed in, use it
              strcpy(user_prompt, cli_user_prompt);
          } else {
              // otherwise get user prompt from stdin
              read_stdin("User: ", user_prompt, sizeof(user_prompt));
          }
          // render user/system prompts into the Llama 2 Chat schema
          if (pos == 0 && system_prompt[0] != '\0') {
              char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
              sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
          } else {
              char user_template[] = "[INST] %s [/INST]";
              sprintf(rendered_prompt, user_template, user_prompt);
          }
          // encode the rendered prompt into tokens
          encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
          user_idx = 0; // reset the user index
          user_turn = 0;
          printf("Assistant: ");
      }

      // determine the token to pass into the transformer next
      if (user_idx < num_prompt_tokens) {
          // if we are still processing the input prompt, force the next prompt token
          token = prompt_tokens[user_idx++];
      } else {
          // otherwise use the next token sampled from previous turn
          token = next;
      }
      // EOS (=2) token ends the Assistant turn
      if (token == 2) { user_turn = 1; }

      // forward the transformer to get logits for the next token
      Tensor *logits = forward(transformer, token, pos);
      next = sample(sampler, logits);
      pos += 1;

      if (user_idx >= num_prompt_tokens && next != 2) {
          // the Assistant is responding, so print its output
          char *piece = decode(tokenizer, token, next);
          safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
          fflush(stdout);
      }
      if (next == 2) { printf("\n"); }
  }
  printf("\n");
  free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING


static void enable_vector_operations() {
unsigned long mstatus;
asm volatile("csrr %0, mstatus" : "=r"(mstatus));
mstatus |= 0x00000600 | 0x00006000 | 0x00018000;
asm volatile("csrw mstatus, %0"::"r"(mstatus));
}

int main() {

enable_vector_operations();


  // default parameters
  float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  int steps = 256;            // number of steps to run for
  char *prompt = NULL;        // prompt string
  unsigned long long rng_seed = 0; // seed rng with time by default
  char *mode = "generate";    // generate|chat
  char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

  // parameter validation/overrides
  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 0;

  // build the Transformer via the model .bin file
  Transformer transformer;
  init_transformer(&transformer);
  if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  init_tokenizer(&tokenizer, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  init_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  printf("Llama 2: a small transformer model for text generation\n");
  // run!
  if (strcmp(mode, "generate") == 0) {
    generate(&transformer, &tokenizer, &sampler, prompt, steps);
  }
  else if (strcmp(mode, "chat") == 0) {
    chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
  }
  else {
    fprintf(stderr, "unknown mode: %s\n", mode);
  }

  // memory and file handles cleanup
  free_sampler(&sampler);
  free_tokenizer(&tokenizer);
  free_transformer(&transformer);
  return 0;
}
#endif