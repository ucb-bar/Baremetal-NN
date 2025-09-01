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


def softmax(x, dim):
    maxes = np.max(x, axis=dim, keepdims=True)
    shifted_exp = np.exp(x - maxes)
    return shifted_exp / shifted_exp.sum(axis=dim, keepdims=True)


def create_sinusoidal_pos_embedding(
    time: float,
    dimension: int,
    min_period: float,
    max_period: float,
) -> np.ndarray:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    fraction = np.linspace(0.0, 1.0, dimension // 2, dtype=np.float64)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * np.pi
    sin_input = scaling_factor * time
    pos_emb = np.concatenate([np.sin(sin_input), np.cos(sin_input)], axis=0)
    pos_emb = pos_emb.astype(np.float32)
    return pos_emb


def eager_attention_forward(
    attention_mask,
    head_dim,
    query_states,
    key_states,
    value_states,
):
    # attention_mask: (51, 867)
    # head_dim: 256
    # query_states: sequence_length, num_att_head, head_dim
    # key_states: sequence_length, num_key_value_head, head_dim
    # value_states: sequence_length, num_key_value_head, head_dim

    # 8
    num_key_value_groups = num_attention_heads // num_key_value_heads
    # 2048
    token_dim = num_key_value_heads * num_key_value_groups * head_dim
    # 816
    sequence_length = query_states.shape[0]

    # (8, 816, 256) <- (816, 8, 256)
    # (8, 51, 256) <- (51, 8, 256)
    query_states = query_states.transpose(1, 0, 2)

    # (8, 816, 256) <- (816, 1, 256)
    # (8, 867, 256) <- (867, 8, 256)
    key_states = np.concatenate([key_states[None, :, 0, :]] * num_key_value_groups, axis=0)
    # (8, 256, 816) <- (8, 816, 256)
    # (8, 256, 867) <- (8, 867, 256)
    key_states_transposed = key_states.transpose(0, 2, 1)

    # (8, 816, 256) <- (816, 1, 256)
    # (8, 867, 256) <- (867, 8, 256)
    value_states = np.concatenate([value_states[None, :, 0, :]] * num_key_value_groups, axis=0)

    # (8, 816, 816) <- (8, 816, 256) x (8, 256, 816)
    att_weights = np.matmul(query_states, key_states_transposed)
    att_weights *= head_dim**-0.5
    big_neg = -2.3819763e38  # See gemma/modules.py

    masked_att_weights = np.where(attention_mask[:, :], att_weights, big_neg)
    probs = softmax(masked_att_weights, dim=-1)

    # (8, 816, 256) <- (8, 816, 816) x (8, 816, 256)
    # (8, 51, 256) <- (51, 867) x (8, 867, 256)
    att_output = np.matmul(probs, value_states)

    # (51, 8, 256) <- (8, 51, 256)
    att_output = att_output.transpose(1, 0, 2)

    # (816, 2048) <- (8, 816, 256)
    # (51, 2048) <- (51, 8, 256)
    att_output = att_output.reshape(sequence_length, token_dim)

    return att_output


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [L,] to x [L, H, D].
    """
    # x: (816, 8, 256)
    d_half = x.shape[-1] // 2

    freq_exponents = (2.0 / x.shape[-1]) * np.arange(d_half)
    timescale = max_wavelength**freq_exponents
    # (816, 128) <- (816, 1) / (1, 128)
    radians = positions[:, None] / timescale[None, :]

    # (816, 1, 128)
    sin = np.sin(radians)[:, None, :]  # .to(dtype=dtype)
    cos = np.cos(radians)[:, None, :]  # .to(dtype=dtype)

    # x1, x2 = x.split(d_half, dim=-1)
    # (816, 8, 128)
    x1 = x[:, :, :d_half]
    x2 = x[:, :, d_half:]
    res = np.empty_like(x)
    res[:, :, :d_half] = x1 * cos - x2 * sin
    res[:, :, d_half:] = x2 * cos + x1 * sin

    return res


def sample_noise(shape):
    noise = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=shape,
    ).astype(np.float32)
    return noise


def make_att_2d_masks(pad_masks, att_masks):
    cumsum = np.cumsum(att_masks, axis=0)
    att_2d_masks = cumsum[None, :] <= cumsum[:, None]
    pad_2d_masks = pad_masks[None, :] * pad_masks[:, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


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

        hidden_shape = (attention_mask.shape[0], -1, layer.self_attn.head_dim)

        # (816, 2048)
        query_states = np.matmul(
            hidden_states,
            layer.self_attn.q_proj.weight.float().detach().cpu().numpy().T,
        )
        # (816, 256)
        key_states = np.matmul(
            hidden_states,
            layer.self_attn.k_proj.weight.float().detach().cpu().numpy().T
        )
        # (816, 256)
        value_states = np.matmul(
            hidden_states,
            layer.self_attn.v_proj.weight.float().detach().cpu().numpy().T,
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

        # (816, 2048) <- (816, 2048) x (2048, 2048)
        out_emb = np.matmul(
            att_output,
            layer.self_attn.o_proj.weight.float().detach().cpu().numpy().T,
        )

        # first residual
        out_emb += hidden_states
        after_first_residual = out_emb.copy()
        # print("out_emb", out_emb[0, :10])

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


def paligemma_action_expert_forward(
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    input_embeds: np.ndarray,
    past_key_values: dict,
):

    model = policy.model.paligemma_with_expert.gemma_expert.model

    # (51, 1024)
    input_embeddings = input_embeds.copy()

    for layer_idx in range(num_hidden_layers):
        # query_states = []
        # key_states = []
        # value_states = []
        layer = model.layers[layer_idx]

        # (51, 1024)
        hidden_states = input_embeddings

        # hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = rms_norm(
            hidden_states,
            layer.input_layernorm.weight.float().detach().cpu().numpy(),
            layer.input_layernorm.eps,
        )

        hidden_shape = (attention_mask.shape[0], -1, layer.self_attn.head_dim)

        # (51, 2048)
        query_states = np.matmul(
            hidden_states,
            layer.self_attn.q_proj.weight.float().detach().cpu().numpy().T
        )
        # (51, 256)
        key_states = np.matmul(
            hidden_states,
            layer.self_attn.k_proj.weight.float().detach().cpu().numpy().T,
        )
        # (51, 256)
        value_states = np.matmul(
            hidden_states,
            layer.self_attn.v_proj.weight.float().detach().cpu().numpy().T
        )

        # (51, 8, 256)
        query_states = query_states.reshape(hidden_shape)
        # (51, 1, 256)
        key_states = key_states.reshape(hidden_shape)
        # (51, 1, 256)
        value_states = value_states.reshape(hidden_shape)

        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)

        key_states = np.concatenate([past_key_values[layer_idx]["key_states"], key_states], axis=0)
        value_states = np.concatenate([past_key_values[layer_idx]["value_states"], value_states], axis=0)

        # (51, 2048)
        att_output = eager_attention_forward(
            attention_mask=attention_mask,
            head_dim=head_dim,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )
        # print("att_output", att_output[0, :10])

        hidden_states = input_embeddings.copy()

        # (51, 1024) <- (51, 2048) x (2048, 1024)
        out_emb = np.matmul(
            att_output,
            layer.self_attn.o_proj.weight.float().detach().cpu().numpy().T,
        )

        # first residual
        out_emb += hidden_states
        after_first_residual = out_emb.copy()
        # print("out_emb", out_emb[0, :10])

        # out_emb = layer.post_attention_layernorm(out_emb)
        out_emb = rms_norm(
            out_emb,
            layer.post_attention_layernorm.weight.float().detach().cpu().numpy(),
            layer.post_attention_layernorm.eps,
        )
        # (51, 1024)
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

    return outputs_embeds


def embed_prefix(
    images, img_masks, lang_tokens, lang_masks
):
    # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
    embs = []
    pad_masks = []
    att_masks = []

    # TODO: remove for loop
    for (
        img,
        img_mask,
    ) in zip(images, img_masks, strict=False):
        # img: [batch, channel, height, width], uint8
        # img_emb: [batch, num_patches, token_dim], float32
        # (256, 2048) <- (3, 224, 224)
        img_emb = policy.model.paligemma_with_expert.embed_image(torch.from_numpy(img[None, ...]).to("cuda"))[0]
        img_emb = img_emb.detach().cpu().numpy()

        # Normalize image embeddings
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * np.array(img_emb_dim**0.5, dtype=img_emb.dtype)

        num_img_embs = img_emb.shape[0]
        # (256,) <- (1,)
        img_mask = np.array([img_mask]).repeat(num_img_embs, axis=0)

        embs.append(img_emb)
        pad_masks.append(img_mask)

        # Create attention masks so that image tokens attend to each other
        att_masks += [0] * num_img_embs

    # lang_tokens: [max_lang_seq_len], int64 --- (48,)
    # lang_emb: [max_lang_seq_len, token_dim], float32 --- (48, 2048)
    lang_emb = policy.model.paligemma_with_expert.embed_language_tokens(torch.from_numpy(lang_tokens).to("cuda"))
    lang_emb = lang_emb.detach().cpu().numpy()

    # Normalize language embeddings
    lang_emb_dim = lang_emb.shape[-1]
    lang_emb = lang_emb * np.sqrt(lang_emb_dim)

    embs.append(lang_emb)
    pad_masks.append(lang_masks)

    # full attention between image and language inputs
    num_lang_embs = lang_emb.shape[0]
    att_masks += [0] * num_lang_embs

    embs = np.concatenate(embs, axis=0)
    pad_masks = np.concatenate(pad_masks, axis=0)
    att_masks = np.array(att_masks, dtype=embs.dtype)

    pad_masks = pad_masks.astype(np.bool)
    att_masks = att_masks.astype(np.bool)
    # att_masks = att_masks[None, :].repeat(bsize, axis=0)

    # embs: [image_seq + lang_seq, token_dim], float32 --- (816, 2048)
    # pad_masks: [image_seq + lang_seq], bool --- (816,)
    #     True for all valid image and language tokens
    # att_masks: [image_seq + lang_seq], bool --- (816,)
    #     all False

    return embs, pad_masks, att_masks


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
    # (1024,) <- (32,) x (32, 1024)
    state_emb = np.matmul(
        state,
        model.state_proj.weight.float().detach().cpu().numpy().T,
    ) + model.state_proj.bias.float().detach().cpu().numpy()
    embs.append(state_emb[None, :])

    state_mask = np.ones((1,), dtype=np.bool)
    pad_masks.append(state_mask)

    # Set attention masks so that image and language inputs do not attend to state or actions
    att_masks += [1]

    # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
    # (1024,)
    time_emb = create_sinusoidal_pos_embedding(
        timestep, proj_width, min_period=4e-3, max_period=4.0
    )
    # print(time_emb[0, :10])

    # Fuse timestep + action information using an MLP
    action_emb = np.matmul(
        noisy_actions,
        model.action_in_proj.weight.float().detach().cpu().numpy().T,
    ) + model.action_in_proj.bias.float().detach().cpu().numpy()

    # (50, 1024) <- (1024,)
    time_emb = time_emb[None, :].repeat(action_emb.shape[0], axis=0)
    # (50, 2048) <- (50, 1024) x (50, 1024)
    action_time_emb = np.concatenate([action_emb, time_emb], axis=1)

    action_time_emb = np.matmul(
        action_time_emb,
        model.action_time_mlp_in.weight.float().detach().cpu().numpy().T,
    ) + model.action_time_mlp_in.bias.float().detach().cpu().numpy()
    # swish == silu
    action_time_emb = torch.nn.functional.silu(torch.from_numpy(action_time_emb)).numpy()
    action_time_emb = np.matmul(
        action_time_emb,
        model.action_time_mlp_out.weight.float().detach().cpu().numpy().T,
    ) + model.action_time_mlp_out.bias.float().detach().cpu().numpy()

    # Add to input tokens
    embs.append(action_time_emb)

    action_time_dim = action_time_emb.shape[0]
    action_time_mask = np.ones((action_time_dim,), dtype=np.bool)
    pad_masks.append(action_time_mask)

    # Set attention masks so that image, language and state inputs do not attend to action tokens
    att_masks += [1] + ([0] * (policy.config.n_action_steps - 1))

    embs = np.concatenate(embs, axis=0)
    pad_masks = np.concatenate(pad_masks, axis=0)
    att_masks = np.array(att_masks, dtype=embs.dtype)
    # att_masks = att_masks[None, :].repeat(bsize, axis=0)

    pad_masks = pad_masks.astype(np.bool)
    att_masks = att_masks.astype(np.bool)

    # embs: [action_seq, token_dim], float32 --- (50, 1024)
    # pad_masks: [action_seq], bool --- (50,)
    # att_masks: [action_seq], bool --- (50,)
    return embs, pad_masks, att_masks


def denoise_step(
    state,
    prefix_pad_masks,
    past_key_values,
    x_t,
    timestep,
):
    """Apply one denoising step of the noise `x_t` at a given timestep."""
    # state: [batch, state_max_dim], float32 --- (32,)
    # prefix_pad_masks: [batch, image_seq + lang_seq], bool --- (816,)
    # past_key_values: per layer kv cache value --- 18 x ((816, 1, 256), (816, 1, 256))
    #   keys are "key_states" and "value_states"
    # x_t: [batch, flow_matching_steps, action_max_dim], float32 --- (50, 32)
    # timestep: [batch], float32 --- (1,)

    # suffix_embs: (51, 1024), float32
    # suffix_pad_masks: (51,), bool
    # suffix_att_masks: (51,), bool
    suffix_embs, suffix_pad_masks, suffix_att_masks = embed_suffix(state, x_t, timestep)

    suffix_len = suffix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[0]

    # (51, 816) <- (816,)
    prefix_pad_2d_masks = prefix_pad_masks[None, :].repeat(suffix_len, axis=0)

    # (51, 51)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    # (51, 867) <- [(51, 816), (51, 51)]
    full_att_2d_masks = np.concatenate([prefix_pad_2d_masks, suffix_att_2d_masks], axis=1)

    prefix_offsets = np.sum(prefix_pad_masks, axis=0)
    # (51,), int64
    position_ids = prefix_offsets + np.cumsum(suffix_pad_masks, axis=0) - 1

    # outputs_embeds: (1, 51, 1024), float32
    outputs_embeds = paligemma_action_expert_forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        input_embeds=suffix_embs,
    )
    suffix_out = outputs_embeds
    # get last n_action_steps: (50, 1024)
    suffix_out = suffix_out[-policy.config.n_action_steps:]

    # (50, 32) <- (50, 1024) x (1024, 32)
    v_t = np.matmul(
        suffix_out,
        policy.model.action_out_proj.weight.float().detach().cpu().numpy().T,
    ) + policy.model.action_out_proj.bias.float().detach().cpu().numpy()

    # v_t: (50, 32), float32
    return v_t


def sample_actions(
    images, img_masks, lang_tokens, lang_masks, state, noise=None,
):
    actions_shape = (policy.config.n_action_steps, policy.config.max_action_dim)
    if noise is None:
        noise = sample_noise(actions_shape)

    prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = np.cumsum(prefix_pad_masks, axis=0) - 1

    _, past_key_values = paligemma_with_expert_forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        input_embeds=prefix_embs,
    )
    # print("KV$:", past_key_values[0]["key_states"][0, 0, -10:])
    # print("state", state[:10])
    # print("noise", noise[0, :10])
    # print("mask:", prefix_pad_masks[:10])

    dt = -1.0 / policy.config.num_steps
    # dt = dt

    x_t = noise
    time = 1.0

    while time >= -dt / 2:
        expanded_time = time

        # v_t: (50, 32), float32
        v_t = denoise_step(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t.copy(),
            expanded_time,
        )
        # print("v_t:", time, v_t[0, :10])

        # Euler step
        x_t += dt * v_t
        time += dt
    return x_t


def select_action(batch):
    """Select a single action given environment observations.

    This method wraps `select_actions` in order to return one action at a time for execution in the
    environment. It works by managing the actions in a queue and only calling `select_actions` when the
    queue is empty.
    """

    batch = normalize_inputs(batch)

    # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
    # querying the policy.
    # images: list of num_img x [batch, channel, height, width], float32 --- 3 x (1, 3, 224, 224)
    # img_masks: list of num_img x [1], bool --- 3 x (1)
    images, img_masks = prepare_images(batch)

    # state: [batch, max_state_dim], float32 --- (1, 32)
    state = prepare_state(batch)

    # lang_tokens: [batch, max_lang_seq_len], int32 --- (1, 48)
    # lang_masks: [batch, max_lang_seq_len], bool --- (1, 48)
    lang_tokens, lang_masks = prepare_language(batch)

    # actions: [batch, steps, max_action_dim], float32 --- (1, 50, 32)
    actions = sample_actions(
        images, img_masks, lang_tokens, lang_masks, state
    )

    # Unpad actions
    original_action_dim = policy.config.action_feature.shape[0]
    # actions: (1, 50, 7)
    actions = actions[:, :, :original_action_dim]

    actions = unnormalize_outputs({"action": actions})["action"]

    # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
    # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
    actions = actions.transpose(0, 1)

    return actions
