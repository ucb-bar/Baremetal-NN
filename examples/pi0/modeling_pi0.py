"""

We follow the original Attention paper and PyTorch to use the following naming convention for dimensions:

- n: layer dimension (number of layers)
- h: number of heads
- e: head dimension (embedding dimension of query, key, value)
- s: sequence length (number of tokens)
- d: hidden dimension / embedding dimension of the hidden layers

@see https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html


"""


import numpy as np

import torch
import torch.nn as nn
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from configuration_pi0 import (
    CONFIG_N_ACTION_STEPS,
    CONFIG_NUM_HIDDEN_LAYERS,
    CONFIG_NUM_HEADS,
    CONFIG_HEAD_DIM,
    CONFIG_NUM_IMAGES,
    CONFIG_IMAGE_PATCH_LEN,
    CONFIG_MAX_LANGUAGE_SEQ_LEN,
    CONFIG_MAX_STATE_DIM,
    CONFIG_MAX_ACTION_DIM,
    CONFIG_VLM_WIDTH,
    CONFIG_VLA_WIDTH,
    CONFIG_VLM_SEQ_LEN,
    CONFIG_VLA_SEQ_LEN,
    CONFIG_TOTAL_SEQ_LEN,
)


# load policy
policy = PI0Policy.from_pretrained("lerobot/pi0", cache_dir="./cache")
policy = policy.float()


# === Basic Functions === #

def pi0_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """
    Args:
        x: input tensor of shape (sequence_length, hidden_dim)
        weight: weight tensor of shape (hidden_dim,)
        eps: a scalar float value to prevent division by zero

    Returns:
        output: (sequence_length, hidden_dim)
    """
    # output = x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))
    # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    # output = output * (1.0 + weight)

    output = x.copy()
    for s in range(x.shape[0]):
        x_rms = np.sqrt(np.sum(x[s, :] * x[s, :]) / x.shape[1] + eps)
        output[s, :] = x[s, :] * (1.0 / x_rms)
        output[s, :] = output[s, :] * (1.0 + weight[:])

    return output


def pi0_softmax(x: np.ndarray) -> np.ndarray:
    """
    Computes the softmax of the input tensor along the **last** dimension (dim=-1).

    Args:
        x: input tensor of shape (num_heads, sequence_length, hidden_dim)

    Returns:
        output: (num_heads, sequence_length, hidden_dim)
    """
    # x_maximum = np.max(x, axis=-1, keepdims=True)
    # shifted_exp = np.exp(x - x_maximum)
    # return shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

    output = x.copy()
    for h in range(x.shape[0]):  # number of heads
        for s in range(x.shape[1]):  # sequence length
            x_maximum = np.max(x[h, s, :])
            shifted_exp = np.exp(x[h, s, :] - x_maximum)
            output[h, s, :] = shifted_exp / shifted_exp.sum()
    return output


def pi0_normalize(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: input tensor of shape (max_state_dim,)

    Returns:
        output: (max_state_dim,)
    """
    mean = np.sum(x) / x.shape[0]
    std = np.sqrt(np.sum(np.power((x - mean), 2)) / x.shape[0])
    x = (x - mean) / (std + 1e-8)
    return x


def pi0_unnormalize(x: np.ndarray) -> np.ndarray:
    """
    Args:
        x: input tensor of shape (max_action_dim,)

    Returns:
        output: (max_action_dim,)
    """
    mean = np.sum(x) / x.shape[0]
    std = np.sqrt(np.sum(np.power((x - mean), 2)) / x.shape[0])
    x = x * std + mean
    return x


def sample_noise(shape: tuple[int, ...]) -> np.ndarray:
    noise = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=shape,
    ).astype(np.float32)
    return noise


# === Data Preprocessing and Tokenization === #

def create_sinusoidal_pos_embedding(
    time: float,
    dimension: int,
    min_period: float,
    max_period: float,
) -> np.ndarray:
    """
    Computes sine-cosine positional embedding vectors for scalar positions.

    Args:
        time: input scalar
        dimension: input dimension
        min_period: input minimum period
        max_period: input maximum period
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    fraction = np.linspace(0.0, 1.0, dimension // 2, dtype=np.float64)
    period = min_period * np.power(max_period / min_period, fraction)

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * np.pi
    sin_input = scaling_factor * time
    pos_emb = np.concatenate([np.sin(sin_input), np.cos(sin_input)], axis=0)
    pos_emb = pos_emb.astype(np.float32)
    return pos_emb


def prepare_images(images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        images: input tensor of shape (num_images, channel, height, width)

    Returns:
        output: tuple consisting of image tensors of shape (num_images, channel, height, width)
            and boolean mask tensor of shape (num_images,)
    """
    num_images = images.shape[0]

    assert images.shape[2] == 224 and images.shape[3] == 224, "images must be resized to 224x224"

    # Preprocess image features present in the batch
    # for key in present_img_keys:
    images[:, :, :, :] = images[:, :, :, :] * 2.0 - 1.0
    img_masks = np.ones(num_images, dtype=np.bool)

    return images, img_masks


def prepare_language(task_text: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        task_text: input string

    Returns:
        output: tuple consisting of language tokens tensor of shape (max_lang_seq_len,)
            and boolean mask tensor of shape (max_lang_seq_len,)
    """

    # PaliGemma prompt has to end with a new line
    # tasks: ["Just try to do something useful.\n"], length of 33
    tasks = [task if task.endswith("\n") else f"{task}\n" for task in task_text]

    # tokenized_prompt["input_ids"]: [batch, max_lang_seq_len], int64 --- (1, 48)
    # tokenized_prompt["attention_mask"]: [batch, max_lang_seq_len], int64 --- (1, 48)
    tokenized_prompt = policy.language_tokenizer.__call__(
        tasks,
        padding="max_length",
        padding_side="right",
        max_length=policy.config.tokenizer_max_length,
        return_tensors="pt",
    )
    lang_tokens = tokenized_prompt["input_ids"][0].detach().cpu().numpy()
    lang_masks = tokenized_prompt["attention_mask"][0].detach().cpu().numpy()

    return lang_tokens, lang_masks


def prepare_state(state: np.ndarray) -> np.ndarray:
    """
    Args:
        state: input tensor of shape (state_dim,)

    Returns:
        output: padded state tensor of shape (max_state_dim,)
    """
    max_state_dim = policy.config.max_state_dim
    state = np.concatenate([state, np.zeros((max_state_dim - state.shape[0],))], axis=0)
    return state


def embed_prefix(
    images: np.ndarray,
    img_masks: np.ndarray,
    lang_tokens: np.ndarray,
    lang_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        images: input tensor of shape (num_images, channel, height, width)
        img_masks: input tensor of shape (num_images,)
        lang_tokens: input tensor of shape (max_lang_seq_len,)
        lang_masks: input tensor of shape (max_lang_seq_len,)

    Returns:
        output: tuple consisting of embedding tensor of shape (image_seq + lang_seq, token_dim),
            boolean mask tensor of shape (image_seq + lang_seq,) and attention mask tensor of
            shape (image_seq + lang_seq,)
    """
    # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
    embs = []
    pad_masks = []
    att_masks = []

    num_images = images.shape[0]

    # TODO: remove for loop
    for img_idx in range(num_images):
        img = images[img_idx, ...]
        img_mask = img_masks[img_idx]
        # img: [batch, channel, height, width], uint8
        # img_emb: [batch, num_patches, token_dim], float32
        # (256, 2048) <- (3, 224, 224)
        img_emb = policy.model.paligemma_with_expert.embed_image(torch.from_numpy(img[None, ...]).to("cuda"))[0]
        img_emb = img_emb.detach().cpu().numpy()

        # Normalize image embeddings
        img_emb_dim = img_emb.shape[-1]
        img_emb = img_emb * np.array(np.sqrt(img_emb_dim), dtype=img_emb.dtype)

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
    state: np.ndarray,
    noisy_actions: np.ndarray,
    timestep: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Args:
        state: input tensor of shape (state_dim,)
        noisy_actions: input tensor of shape (num_actions, action_dim)
        timestep: input scalar

    Returns:
        output: tuple consisting of embedding tensor of shape (num_actions, token_dim),
            boolean mask tensor of shape (num_actions,) and attention mask tensor of shape (num_actions,)
    """
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
    # (50, 1024) <- (50, 32) x (32, 1024)
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


# === Transformer Components === #

def make_att_2d_masks(pad_masks: np.ndarray, att_masks: np.ndarray) -> np.ndarray:
    cumsum = np.cumsum(att_masks, axis=0)
    att_2d_masks = cumsum[None, :] <= cumsum[:, None]
    pad_2d_masks = pad_masks[None, :] * pad_masks[:, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def apply_rope(x, positions: np.ndarray, max_wavelength: float = 10_000) -> np.ndarray:
    """
    Applies RoPE position encoding.

    Args:
        x: input tensor of shape (sequence_length, num_heads, head_dim)
        positions: input tensor of shape (sequence_length,)
        max_wavelength: input maximum wavelength

    Returns:
        output: (sequence_length, num_heads, head_dim)
    """
    # x: (816, 8, 256)
    head_dim = x.shape[2]
    half_head_dim = head_dim // 2

    freq_exponents = (2.0 / head_dim) * np.arange(half_head_dim)
    timescale = np.power(max_wavelength, freq_exponents)
    # (816, 128) <- (816, 1) / (1, 128)
    radians = positions[:, None] / timescale[None, :]

    # (816, 1, 128)
    sin = np.sin(radians)[:, None, :]  # .to(dtype=dtype)
    cos = np.cos(radians)[:, None, :]  # .to(dtype=dtype)

    # (816, 8, 128)
    x1 = x[:, :, :half_head_dim]
    x2 = x[:, :, half_head_dim:]
    res = np.empty_like(x)
    res[:, :, :half_head_dim] = x1 * cos - x2 * sin
    res[:, :, half_head_dim:] = x2 * cos + x1 * sin

    return res


def eager_attention_forward(
    attention_mask: np.ndarray,
    query_states: np.ndarray,
    key_states: np.ndarray,
    value_states: np.ndarray,
) -> np.ndarray:
    """
    Attention forward pass.

    Args:
        attention_mask: input tensor of shape (sequence_length, sequence_length)
        query_states: input tensor of shape (sequence_length, num_heads, head_dim)
        key_states: input tensor of shape (sequence_length, 1, head_dim)
        value_states: input tensor of shape (sequence_length, 1, head_dim)

    Returns:
        output: (sequence_length, hidden_dim)
    """

    # vlm: 816
    # acs: 51
    sequence_length = query_states.shape[0]
    # 8
    num_heads = query_states.shape[1]
    # 256
    head_dim = query_states.shape[2]
    # 2048
    hidden_dim = num_heads * head_dim

    # vlm: (8, 816, 256) <- (816, 8, 256)
    # acs: (8, 51, 256) <- (51, 8, 256)
    query_states = query_states.transpose(1, 0, 2)

    # vlm: (8, 816, 256) <- (816, 1, 256)
    # acs: (8, 867, 256) <- (867, 1, 256)
    key_states = np.concatenate([key_states[None, :, 0, :]] * num_heads, axis=0)

    # vlm: (8, 816, 256) <- (816, 1, 256)
    # acs: (8, 867, 256) <- (867, 1, 256)
    value_states = np.concatenate([value_states[None, :, 0, :]] * num_heads, axis=0)

    # vlm: (8, 816, 816) <- (8, 816, 256) x (8, 256, 816)
    # acs: (8, 867, 867) <- (8, 867, 256) x (8, 256, 867)
    key_states_transposed = key_states.transpose(0, 2, 1)
    att_weights = np.matmul(query_states, key_states_transposed)

    att_weights *= 1 / np.sqrt(head_dim)
    big_neg = -2.3819763e38  # See gemma/modules.py

    masked_att_weights = np.where(attention_mask[:, :], att_weights, big_neg)
    probs = pi0_softmax(masked_att_weights)

    # vlm: (8, 816, 256) <- (816, 816) x (8, 816, 256)
    # acs: (8, 51, 256) <- (51, 867) x (8, 867, 256)
    att_output = np.matmul(probs, value_states)

    # vlm: (51, 8, 256) <- (8, 51, 256)
    # acs: (51, 8, 256) <- (8, 51, 256)
    att_output = att_output.transpose(1, 0, 2)

    # vlm: (816, 2048) <- (816, 8, 256)
    # acs: (51, 2048) <- (51, 8, 256)
    att_output = att_output.reshape(sequence_length, hidden_dim)

    return att_output


def gemma_mlp_forward(layer: nn.Module, x: np.ndarray) -> np.ndarray:
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


# === Model Forward Functions === #

def paligemma_vlm_forward(
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    input_embeddings: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """
    Args:
        attention_mask: input tensor of shape (sequence_length, sequence_length)
        position_ids: input tensor of shape (sequence_length,)
        input_embeddings: input tensor of shape (sequence_length, hidden_dim)

    Returns:
        output: output embeddings of shape (sequence_length, hidden_dim)
        past_key_values: input dictionary of shape (num_hidden_layers, {key_states, value_states}), where
            key_states and value_states are tensors of shape (sequence_length, 1, head_dim)
    """
    model = policy.model.paligemma_with_expert.paligemma.language_model

    # (816, 2048)
    hidden_states = np.zeros((CONFIG_VLM_SEQ_LEN, CONFIG_VLM_WIDTH))
    hidden_states_residual = np.zeros((CONFIG_VLM_SEQ_LEN, CONFIG_VLM_WIDTH))
    hidden_states[:, :] = input_embeddings

    past_key_values = {}

    for layer_idx in range(CONFIG_NUM_HIDDEN_LAYERS):
        layer = model.layers[layer_idx]

        hidden_shape = (CONFIG_VLM_SEQ_LEN, -1, CONFIG_HEAD_DIM)

        # (816, 2048)
        hidden_states_residual[:, :] = hidden_states

        # LayerNorm
        hidden_states[:, :] = pi0_rms_norm(
            hidden_states,
            layer.input_layernorm.weight.float().detach().cpu().numpy(),
            layer.input_layernorm.eps,
        )

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

        query_states[:, :] = apply_rope(query_states, position_ids)
        key_states[:, :] = apply_rope(key_states, position_ids)

        # store KV cache
        past_key_values[layer_idx] = {
            "key_states": key_states,
            "value_states": value_states,
        }

        # (816, 2048)
        hidden_states[:, :] = eager_attention_forward(
            attention_mask=attention_mask,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )
        # (816, 2048) <- (816, 2048) x (2048, 2048)
        hidden_states[:, :] = np.matmul(
            hidden_states,
            layer.self_attn.o_proj.weight.float().detach().cpu().numpy().T,
        )

        # first residual
        hidden_states[:, :] += hidden_states_residual
        hidden_states_residual[:, :] = hidden_states

        # LayerNorm
        hidden_states[:, :] = pi0_rms_norm(
            hidden_states,
            layer.post_attention_layernorm.weight.float().detach().cpu().numpy(),
            layer.post_attention_layernorm.eps,
        )
        # (816, 2048)
        hidden_states[:, :] = gemma_mlp_forward(layer.mlp, hidden_states)

        # second residual
        hidden_states[:, :] += hidden_states_residual

    # final norm
    hidden_states[:, :] = pi0_rms_norm(
        hidden_states,
        model.norm.weight.float().detach().cpu().numpy(),
        model.norm.eps,
    )

    return hidden_states, past_key_values


def paligemma_action_forward(
    attention_mask: np.ndarray,
    position_ids: np.ndarray,
    input_embeddings: np.ndarray,
    past_key_values: dict,
) -> np.ndarray:
    """
    Args:
        attention_mask: input tensor of shape (sequence_length, sequence_length)
        position_ids: input tensor of shape (sequence_length,)
        input_embeddings: input tensor of shape (sequence_length, hidden_dim)
        past_key_values: input dictionary of shape (num_hidden_layers, {key_states, value_states}), where
            key_states and value_states are tensors of shape (sequence_length, 1, head_dim)

    Returns:
        output: output embeddings of shape (sequence_length, hidden_dim)
    """
    model = policy.model.paligemma_with_expert.gemma_expert.model

    # (51, 1024)
    hidden_states = np.zeros((CONFIG_VLA_SEQ_LEN, CONFIG_VLA_WIDTH))
    hidden_states_residual = np.zeros((CONFIG_VLA_SEQ_LEN, CONFIG_VLA_WIDTH))
    attention_output = np.zeros((CONFIG_VLA_SEQ_LEN, CONFIG_NUM_HEADS * CONFIG_HEAD_DIM))
    hidden_states[:, :] = input_embeddings

    for layer_idx in range(CONFIG_NUM_HIDDEN_LAYERS):
        layer = model.layers[layer_idx]

        hidden_shape = (CONFIG_VLA_SEQ_LEN, -1, CONFIG_HEAD_DIM)

        # (51, 1024)
        hidden_states_residual[:, :] = hidden_states

        # LayerNorm
        hidden_states[:, :] = pi0_rms_norm(
            hidden_states,
            layer.input_layernorm.weight.float().detach().cpu().numpy(),
            layer.input_layernorm.eps,
        )

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

        query_states[:, :] = apply_rope(query_states, position_ids)
        key_states[:, :] = apply_rope(key_states, position_ids)

        # join key_states and value_states from KV cache
        key_states = np.concatenate([past_key_values[layer_idx]["key_states"], key_states], axis=0)
        value_states = np.concatenate([past_key_values[layer_idx]["value_states"], value_states], axis=0)

        # (51, 2048)
        attention_output[:, :] = eager_attention_forward(
            attention_mask=attention_mask,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )
        # (51, 1024) <- (51, 2048) x (2048, 1024)
        hidden_states[:, :] = np.matmul(
            attention_output,
            layer.self_attn.o_proj.weight.float().detach().cpu().numpy().T,
        )

        # first residual
        hidden_states[:, :] += hidden_states_residual
        hidden_states_residual[:, :] = hidden_states

        # LayerNorm
        hidden_states[:, :] = pi0_rms_norm(
            hidden_states,
            layer.post_attention_layernorm.weight.float().detach().cpu().numpy(),
            layer.post_attention_layernorm.eps,
        )
        # (51, 1024)
        hidden_states[:, :] = gemma_mlp_forward(layer.mlp, hidden_states)

        # second residual
        hidden_states[:, :] += hidden_states_residual

    # final norm
    hidden_states[:, :] = pi0_rms_norm(
        hidden_states,
        model.norm.weight.float().detach().cpu().numpy(),
        model.norm.eps,
    )

    return hidden_states


# === Larger Building Blocks === #

def denoise_step(
    state: np.ndarray,
    prefix_pad_masks: np.ndarray,
    past_key_values: dict[int, dict[str, np.ndarray]],
    x_t: np.ndarray,
    timestep: float,
) -> np.ndarray:
    """
    Apply one denoising step of the noise `x_t` at a given timestep.

    Args:
        state: input tensor of shape (state_max_dim,)
        prefix_pad_masks: input tensor of shape (image_sequence_length + lang_sequence_length,) with boolean type
        past_key_values: input dictionary of shape (num_hidden_layers, {key_states, value_states}), where
            key_states and value_states are tensors of shape (sequence_length, 1, head_dim)
        x_t: input tensor of shape (flow_matching_steps, action_max_dim)
        timestep: input scalar

    Returns:
        output tensor of shape (flow_matching_steps, action_max_dim)
    """
    # state: (32,)
    # prefix_pad_masks: (816,)
    # past_key_values: 18 x {(key_states: (816, 1, 256), value_states: (816, 1, 256))}
    # x_t: (50, 32)
    # timestep: 1.0

    # (51, 1024) (51,) (51,)
    suffix_embs, suffix_pad_masks, suffix_att_masks = embed_suffix(state, x_t, timestep)

    suffix_len = suffix_pad_masks.shape[0]

    # (51, 816) <- (816,)
    prefix_pad_2d_masks = prefix_pad_masks[None, :].repeat(suffix_len, axis=0)

    # (51, 51)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

    # (51, 867) <- [(51, 816), (51, 51)]
    full_att_2d_masks = np.concatenate([prefix_pad_2d_masks, suffix_att_2d_masks], axis=1)

    prefix_offsets = np.sum(prefix_pad_masks, axis=0)
    # (51,)
    position_ids = prefix_offsets + np.cumsum(suffix_pad_masks, axis=0) - 1

    # (1, 51, 1024)
    outputs_embeds = paligemma_action_forward(
        attention_mask=full_att_2d_masks,
        position_ids=position_ids,
        past_key_values=past_key_values,
        input_embeddings=suffix_embs,
    )
    suffix_out = outputs_embeds
    # get last n_action_steps: (50, 1024)
    suffix_out = suffix_out[-policy.config.n_action_steps:]

    # (50, 32) <- (50, 1024) x (1024, 32)
    v_t = np.matmul(
        suffix_out,
        policy.model.action_out_proj.weight.float().detach().cpu().numpy().T,
    ) + policy.model.action_out_proj.bias.float().detach().cpu().numpy()

    # v_t: (50, 32)
    return v_t


def sample_actions(
    images: np.ndarray,
    img_masks: np.ndarray,
    lang_tokens: np.ndarray,
    lang_masks: np.ndarray,
    state: np.ndarray,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        images: input tensor of shape (num_images, channel, height, width)
        img_masks: input tensor of shape (num_images,) with boolean type
        lang_tokens: input tensor of shape (max_lang_seq_len,)
        lang_masks: input tensor of shape (max_lang_seq_len,) with boolean type
        state: input tensor of shape (state_max_dim,)
        noise: input tensor of shape (n_action_steps, max_action_dim)

    Returns:
        output tensor of shape (n_action_steps, max_action_dim)
    """
    actions_shape = (policy.config.n_action_steps, policy.config.max_action_dim)
    if noise is None:
        noise = sample_noise(actions_shape)

    prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = np.cumsum(prefix_pad_masks, axis=0) - 1

    _, past_key_values = paligemma_vlm_forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        input_embeddings=prefix_embs,
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


def select_action(
    images: np.ndarray,
    state: np.ndarray,
    task: str,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """
    Args:
        images: input tensor of shape (num_images, channel, height, width)
        state: input tensor of shape (state_max_dim,)
        task: string that describes the task
        noise: input tensor of shape (n_action_steps, max_action_dim)

    Returns:
        output tensor of shape (n_action_steps, max_action_dim)
    """
    # HACK: match hf policy
    # state = pi0_normalize(state)

    # (3, 3, 224, 224) (3,)
    images, img_masks = prepare_images(images)

    # (32,)
    state = prepare_state(state)

    # (48,) (48,)
    lang_tokens, lang_masks = prepare_language(task)

    # print("images", images[0, 0, 0, :10])
    # print("lang_tokens", lang_tokens[:10])
    # print("state", state[:10])

    # (50, 32)
    actions = sample_actions(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    )
    # print("actions", actions[0, :10])

    # unpad actions
    original_action_dim = 7
    # (50, 7) <- (50, 32)
    actions = actions[:, :original_action_dim]

    # HACK: match hf policy
    # actions = unnormalize_actions(actions)

    return actions[0]
