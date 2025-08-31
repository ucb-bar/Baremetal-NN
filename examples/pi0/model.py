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


def create_sinusoidal_pos_embedding(
    time: np.ndarray, dimension: int, min_period: float, max_period: float,
) -> np.ndarray:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = np.linspace(0.0, 1.0, dimension // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * np.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = np.concatenate([np.sin(sin_input), np.cos(sin_input)], axis=1)
    pos_emb = pos_emb.astype(np.float32)
    return pos_emb


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


def embed_prefix(
    images, img_masks, lang_tokens, lang_masks,
):
    return policy.model.paligemma_with_expert.paligemma.language_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )


def embed_suffix(
    state,
    noisy_actions,
    timestep
):
    embs = []
    pad_masks = []
    att_masks = []

    proj_width = 1024

    model = policy.model

    # Embed state
    state_emb = np.matmul(state, model.state_proj.weight.float().detach().cpu().numpy().T)
    embs.append(state_emb[:, None, :])
    bsize = state_emb.shape[0]

    state_mask = np.ones((bsize, 1), dtype=np.bool)
    pad_masks.append(state_mask)

    # Set attention masks so that image and language inputs do not attend to state or actions
    att_masks += [1]

    # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
    time_emb = create_sinusoidal_pos_embedding(
        timestep, proj_width, min_period=4e-3, max_period=4.0
    )
    # print(time_emb[0, :10])

    # Fuse timestep + action information using an MLP
    action_emb = np.matmul(noisy_actions, model.action_in_proj.weight.float().detach().cpu().numpy().T) + model.action_in_proj.bias.float().detach().cpu().numpy()

    time_emb = time_emb[:, None, :].repeat(action_emb.shape[1], axis=1)
    action_time_emb = np.concatenate([action_emb, time_emb], axis=2)

    action_time_emb = np.matmul(action_time_emb, model.action_time_mlp_in.weight.float().detach().cpu().numpy().T) + model.action_time_mlp_in.bias.float().detach().cpu().numpy()
    # swish == silu
    action_time_emb = torch.nn.functional.silu(torch.from_numpy(action_time_emb)).numpy()
    action_time_emb = np.matmul(action_time_emb, model.action_time_mlp_out.weight.float().detach().cpu().numpy().T) + model.action_time_mlp_out.bias.float().detach().cpu().numpy()

    # Add to input tokens
    embs.append(action_time_emb)

    bsize, action_time_dim = action_time_emb.shape[:2]
    action_time_mask = np.ones((bsize, action_time_dim), dtype=np.bool)
    pad_masks.append(action_time_mask)

    # Set attention masks so that image, language and state inputs do not attend to action tokens
    att_masks += [1] + ([0] * (policy.config.n_action_steps - 1))

    embs = np.concatenate(embs, axis=1)
    pad_masks = np.concatenate(pad_masks, axis=1)
    att_masks = np.array(att_masks, dtype=embs.dtype)
    att_masks = att_masks[None, :].repeat(bsize, axis=0)

    return embs, pad_masks, att_masks


def denoise_step(
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    suffix_embs, suffix_pad_masks, suffix_att_masks = embed_suffix(state, x_t, timestep)




def sample_actions(
    images, img_masks, lang_tokens, lang_masks, state,
):
    prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

    _, past_key_values = paligemma_with_expert_forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        inputs_embeds=prefix_embs,
    )
    
    dt = -1.0 / self.config.num_steps
    dt = torch.tensor(dt, dtype=torch.float32, device=device)

    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)

    while time >= -dt / 2:
        expanded_time = time.expand(bsize)

        # v_t: (1, 50, 32), float32
        v_t = self.denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            expanded_time,
        )

        # Euler step
        x_t += dt * v_t
        time += dt
    return x_t
