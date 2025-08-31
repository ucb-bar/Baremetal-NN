import numpy as np

import torch
from lerobot.policies.pi0.modeling_pi0 import PI0Policy


num_hidden_layers = 18
num_attention_heads = 8
num_key_value_heads = 1
head_dim = 256


# load policy
policy = PI0Policy.from_pretrained("lerobot/pi0", cache_dir="./cache")
policy = policy.float()


def rms_norm(x, weight, eps):
    output = x * (1.0 / np.sqrt(np.power(x, 2).mean(-1, keepdims=True) + eps))
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + weight)
    return output


def softmax(x, dim=-1):
    maxes = np.max(x, axis=dim, keepdims=True)
    shifted_exp = np.exp(x - maxes)
    return shifted_exp / shifted_exp.sum(axis=dim, keepdims=True)


def eager_attention_forward(
    attention_mask, head_dim, query_states, key_states, value_states
):
    # query_states: sequence_length, num_att_head, head_dim
    # key_states: sequence_length, num_key_value_head, head_dim
    # value_states: sequence_length, num_key_value_head, head_dim

    # 8
    num_key_value_groups = num_attention_heads // num_key_value_heads
    # 2048
    token_dim = num_key_value_heads * num_key_value_groups * head_dim
    # 816
    sequence_length = key_states.shape[0]

    # (8, 816, 256) <- (816, 8, 256)
    query_states = query_states.transpose(1, 0, 2)

    # (8, 816, 256) <- (816, 1, 256)
    key_states = np.concatenate([key_states[None, :, 0, :]] * num_key_value_groups, axis=0)
    # (8, 256, 816) <- (8, 816, 256)
    key_states_transposed = key_states.transpose(0, 2, 1)

    # (8, 816, 256) <- (816, 1, 256)
    value_states = np.concatenate([value_states[None, :, 0, :]] * num_key_value_groups, axis=0)

    # (8, 816, 816) <- (8, 816, 256) x (8, 256, 816)
    att_weights = np.matmul(query_states, key_states_transposed)
    att_weights *= head_dim**-0.5
    big_neg = -2.3819763e38  # See gemma/modules.py

    masked_att_weights = np.where(attention_mask[:, :], att_weights, big_neg)
    probs = softmax(masked_att_weights, dim=-1)

    # (8, 816, 256) <- (8, 816, 816) x (8, 816, 256)
    att_output = np.matmul(probs, value_states)

    # (816, 8, 256) <- (8, 816, 256)
    att_output = att_output.transpose(1, 0, 2)

    # (816, 2048) <- (8, 816, 256)
    att_output = att_output.reshape(sequence_length, token_dim)

    return att_output


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2

    freq_exponents = (2.0 / x.shape[-1]) * np.arange(d_half)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]

    radians = radians[..., None, :]

    sin = np.sin(radians)  # .to(dtype=dtype)
    cos = np.cos(radians)  # .to(dtype=dtype)

    # x1, x2 = x.split(d_half, dim=-1)
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    res = np.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res


def gemma_mlp_forward(layer, x):
    gate_proj = layer.gate_proj
    up_proj = layer.up_proj
    down_proj = layer.down_proj

    x_gated = np.matmul(
        x,
        gate_proj.weight.float().detach().cpu().numpy().T
    )
    x_gated_activated = torch.nn.functional.gelu(torch.from_numpy(x_gated), approximate="tanh").numpy()
    x_up_projected = np.matmul(
        x,
        up_proj.weight.float().detach().cpu().numpy().T
    )
    x_up = x_up_projected * x_gated_activated
    x_down_projected = np.matmul(
        x_up,
        down_proj.weight.float().detach().cpu().numpy().T
    )
    return x_down_projected


def paligemma_with_expert_forward(
    attention_mask: np.ndarray,
    position_ids: np.ndarray | None = None,
    input_embeds: np.ndarray = None,
):

    model = policy.model.paligemma_with_expert.paligemma.language_model

    # (816, 2048)
    input_embeddings = input_embeds.copy()
    past_key_values = {}

    for layer_idx in range(num_hidden_layers):
        # query_states = []
        # key_states = []
        # value_states = []
        layer = model.layers[layer_idx]

        # (816, 2048)
        hidden_states = input_embeddings

        # hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = rms_norm(
            hidden_states,
            layer.input_layernorm.weight.float().detach().cpu().numpy(),
            layer.input_layernorm.eps,
        )

        hidden_shape = (hidden_states.shape[0], -1, layer.self_attn.head_dim)

        # (816, 2048)
        query_states = np.matmul(
            hidden_states,
            layer.self_attn.q_proj.weight.float().detach().cpu().numpy().T
        )
        # (816, 256)
        key_states = np.matmul(
            hidden_states,
            layer.self_attn.k_proj.weight.float().detach().cpu().numpy().T
        )
        # (816, 256)
        value_states = np.matmul(
            hidden_states,
            layer.self_attn.v_proj.weight.float().detach().cpu().numpy().T
        )

        # (816, 8, 256)
        query_states = query_states.reshape(hidden_shape)
        # (816, 1, 256)
        key_states = key_states.reshape(hidden_shape)
        # (816, 1, 256)
        value_states = value_states.reshape(hidden_shape)

        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)

        past_key_values[layer_idx] = {
            "key_states": key_states,
            "value_states": value_states,
        }

        # (816, 2048)
        att_output = eager_attention_forward(
            attention_mask=attention_mask,
            head_dim=head_dim,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )
        # print("att_output", att_output[0, 0, :10])

        hidden_states = input_embeddings.copy()

        start = 0
        end = start + hidden_states.shape[1]
        # (816, 2048) <- (816, 2048) x (2048, 2048)
        out_emb = np.matmul(
            att_output[:, start:end],
            layer.self_attn.o_proj.weight.float().detach().cpu().numpy().T
        )

        # first residual
        out_emb += hidden_states
        after_first_residual = out_emb.copy()
        # print("out_emb", out_emb[0, 0, :10])

        # out_emb = layer.post_attention_layernorm(out_emb)
        out_emb = rms_norm(
            out_emb,
            layer.post_attention_layernorm.weight.float().detach().cpu().numpy(),
            layer.post_attention_layernorm.eps,
        )
        # (816, 2048)
        out_emb = gemma_mlp_forward(layer.mlp, out_emb)

        # second residual
        out_emb += after_first_residual

        outputs_embeds = out_emb

        input_embeddings = outputs_embeds

    # final norm
    # out_emb = model.norm(hidden_states)
    outputs_embeds = rms_norm(
        input_embeddings,
        model.norm.weight.float().detach().cpu().numpy(),
        model.norm.eps,
    )

    return outputs_embeds, past_key_values
