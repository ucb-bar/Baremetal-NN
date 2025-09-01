import numpy as np
import torch
from lerobot.policies.pi0.paligemma_with_expert import PaliGemmaWithExpertModel, PaliGemmaWithExpertConfig
from lerobot.policies.pi0.modeling_pi0 import create_sinusoidal_pos_embedding as create_sinusoidal_pos_embedding_golden
from lerobot.policies.pi0.modeling_pi0 import make_att_2d_masks as make_att_2d_masks_golden
from lerobot.configs.types import FeatureType, PolicyFeature
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm

from modeling_pi0 import (
    eager_attention_forward,
    sample_noise,
    make_att_2d_masks,
    gemma_mlp_forward,
    create_sinusoidal_pos_embedding,
    embed_prefix,
    embed_suffix,
    sample_actions,
    policy,
    paligemma_vlm_forward,
    paligemma_action_forward,
    denoise_step,
    select_action,
    pi0_rms_norm,
)


# initialize policy config
policy.config.empty_cameras = 3  # support up to 3 camera inputs
policy.config.input_features = {
    "observation.images.top": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224)
    ),
    "observation.images.left_wrist": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224)
    ),
    "observation.images.right_wrist": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 224, 224)
    ),
    "observation.state": PolicyFeature(
        type=FeatureType.STATE,
        shape=(7,)
    )
}
policy.config.output_features = {
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(7,)
    )
}


test_past_key_values = {}
test_past_key_values_torch = {}
for layer_idx in range(18):
    key_states = np.random.randn(816, 1, 256).astype(np.float32)
    value_states = np.random.randn(816, 1, 256).astype(np.float32)
    test_past_key_values[layer_idx] = {
        "key_states": key_states,
        "value_states": value_states,
    }
    test_past_key_values_torch[layer_idx] = {
        "key_states": torch.from_numpy(key_states[None, ...]).to("cuda"),
        "value_states": torch.from_numpy(value_states[None, ...]).to("cuda"),
    }


def test_rms_norm():
    print("TEST test_rms_norm")
    test_x = np.random.randn(816, 256).astype(np.float32)
    test_weight = np.random.randn(256,).astype(np.float32)
    test_eps = 1e-5
    test_rms_norm = pi0_rms_norm(test_x, test_weight, test_eps)
    torch_rms = GemmaRMSNorm(dim=test_weight.shape[0], eps=test_eps)
    torch_rms.weight.data = torch.from_numpy(test_weight)
    test_rms_norm_golden = torch_rms(torch.from_numpy(test_x))
    if np.allclose(test_rms_norm, test_rms_norm_golden.float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test_rms_norm:", test_rms_norm[0, :10])
        print("golden test_rms_norm:", test_rms_norm_golden[0, :10])


def test_sample_noise():
    print("TEST test_sample_noise")
    test_shape = (816, 256)
    test_noise = sample_noise(test_shape)
    test_noise_golden = policy.model.sample_noise(
        (1, *test_shape),
        torch.device("cuda"),
    ).float().detach().cpu().numpy()
    # if shape matches and means and stds match
    if (
        test_noise.shape == test_noise_golden.shape[1:]
        and np.abs(np.mean(test_noise) - np.mean(test_noise_golden)) < 0.1
        and np.abs(np.std(test_noise) - np.std(test_noise_golden)) < 0.1
    ):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test_noise:", test_noise.shape, np.mean(test_noise), np.std(test_noise))
        print("golden test_noise:", test_noise_golden.shape, np.mean(test_noise_golden), np.std(test_noise_golden))


def tset_create_sinusoidal_pos_embedding():
    print("TEST test_create_sinusoidal_pos_embedding")
    test_timestep = 0.5
    test_proj_width = 2048

    time_emb = create_sinusoidal_pos_embedding(
        test_timestep,
        test_proj_width,
        min_period=4e-3,
        max_period=4.0,
    )
    time_emb_golden = create_sinusoidal_pos_embedding_golden(
        torch.tensor([test_timestep]).to("cuda"),
        test_proj_width,
        min_period=4e-3,
        max_period=4.0,
        device=torch.device("cuda"),
    )

    if np.allclose(time_emb, time_emb_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("time_emb:", time_emb[0, :10])
        print("golden time_emb:", time_emb_golden[0][0, 0, :10])


def test_embed_prefix():
    print("TEST test_embed_prefix")
    test_images = np.random.randn(3, 3, 224, 224).astype(np.float32)
    test_img_masks = np.ones((3,), dtype=np.bool)
    test_lang_tokens = np.random.randint(0, 100, (48,)).astype(np.int32)
    test_lang_masks = np.ones((48,), dtype=np.bool)

    prefix_embs, prefix_pad_masks, prefix_att_masks = embed_prefix(
        test_images,
        test_img_masks,
        test_lang_tokens,
        test_lang_masks,
    )
    prefix_embs_golden, prefix_pad_masks_golden, prefix_att_masks_golden = policy.model.embed_prefix(
        [torch.from_numpy(test_images[img_idx, ...][None, ...]).to("cuda") for img_idx in range(test_images.shape[0])],
        [torch.tensor([1 if mask else 0], dtype=torch.bool).to("cuda") for mask in test_img_masks],
        torch.from_numpy(test_lang_tokens[None, ...]).to("cuda"),
        torch.from_numpy(test_lang_masks[None, ...]).to("cuda"),
    )
    if np.allclose(prefix_embs, prefix_embs_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("prefix_embs:", prefix_embs[0, :10])
        print("golden prefix_embs:", prefix_embs_golden[0, 0, :10])


def test_embed_suffix():
    print("TEST test_embed_suffix")
    test_state = np.random.randn(32,).astype(np.float32)
    test_noisy_actions = np.random.randn(50, 32).astype(np.float32)
    test_timestep = 0.5

    suffix_embs, suffix_pad_masks, suffix_att_masks = embed_suffix(
        test_state,
        test_noisy_actions,
        test_timestep,
    )
    suffix_embs_golden, suffix_pad_masks_golden, suffix_att_masks_golden = policy.model.embed_suffix(
        torch.from_numpy(test_state[None, ...]).to("cuda"),
        torch.from_numpy(test_noisy_actions[None, ...]).to("cuda"),
        torch.tensor([test_timestep]).to("cuda"),
    )
    if np.allclose(suffix_embs, suffix_embs_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("suffix_embs:", suffix_embs[0, :10])
        print("golden suffix_embs:", suffix_embs_golden[0, 0, :10])


def test_make_att_2d_masks():
    print("TEST test_make_att_2d_masks")
    test_pad_masks = np.ones((816,), dtype=np.bool)
    test_att_masks = np.ones((816,), dtype=np.bool)
    test_att_2d_masks = make_att_2d_masks(
        test_pad_masks,
        test_att_masks,
    )
    test_att_2d_masks_golden = make_att_2d_masks_golden(
        torch.from_numpy(test_pad_masks[None, ...]).to("cuda"),
        torch.from_numpy(test_att_masks[None, ...]).to("cuda"),
    )
    if np.allclose(test_att_2d_masks, test_att_2d_masks_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test_att_2d_masks:", test_att_2d_masks[0, :10])
        print("golden test_att_2d_masks:", test_att_2d_masks_golden[0, 0, :10])


def test_attention():
    print("TEST test_attention")
    test_attention_mask = np.tril(np.ones((816,), dtype=np.bool))
    test_query_states = np.random.randn(816, 8, 256).astype(np.float32)
    test_key_states = np.random.randn(816, 1, 256).astype(np.float32)
    test_value_states = np.random.randn(816, 1, 256).astype(np.float32)

    # our implementation does not support batching
    att_output = eager_attention_forward(
        test_attention_mask,
        test_query_states,
        test_key_states,
        test_value_states,
    )

    model = PaliGemmaWithExpertModel(PaliGemmaWithExpertConfig())

    att_output_golden = model.eager_attention_forward(
        attention_mask=torch.from_numpy(test_attention_mask[None, ...]),
        batch_size=1,
        head_dim=test_query_states.shape[-1],
        query_states=torch.from_numpy(test_query_states[None, ...]),
        key_states=torch.from_numpy(test_key_states[None, ...]),
        value_states=torch.from_numpy(test_value_states[None, ...]),
    )
    if np.allclose(att_output, att_output_golden, atol=1e-4, rtol=1e-4):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test att_output:", att_output[0, :10])
        print("golden att_output:", att_output_golden[0, 0, :10])


def test_vlm_mlp_forward():
    print("TEST test_vlm_mlp_forward")
    # Test MLP forward
    test_input_emb = np.random.randn(1, 816, 2048).astype(np.float32)

    model = policy.model.paligemma_with_expert.paligemma.language_model
    # our implementation does not support batching
    out_emb = gemma_mlp_forward(model.layers[0].mlp, test_input_emb[0, ...])

    out_emb_gold = policy.model.paligemma_with_expert.paligemma.language_model.layers[0].mlp(
        torch.from_numpy(test_input_emb).to("cuda")
    )

    if np.allclose(out_emb, out_emb_gold.float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test out_emb:", out_emb[0, :10])
        print("golden out_emb:", out_emb_gold[0, 0, :10])


def test_vlm_forward():
    print("TEST test_vlm_forward")
    test_attention_2d_mask = np.tril(np.ones((816,), dtype=np.bool))
    test_position_ids = np.arange(816, dtype=np.int64)
    test_prefix_embs = np.random.randn(816, 2048).astype(np.float32)

    att_output, kv_cache_output = paligemma_vlm_forward(
        attention_mask=test_attention_2d_mask,
        position_ids=test_position_ids,
        input_embeddings=test_prefix_embs,
    )

    att_output_golden, kv_cache_output_golden = policy.model.paligemma_with_expert.forward(
        attention_mask=torch.from_numpy(test_attention_2d_mask[None, ...]).to("cuda"),
        position_ids=torch.from_numpy(test_position_ids[None, ...]).to("cuda"),
        past_key_values=None,
        inputs_embeds=[torch.from_numpy(test_prefix_embs[None, ...]).to("cuda"), None],
        use_cache=True,
        fill_kv_cache=True,
    )

    kv_cache_match = True
    for layer_idx in range(18):
        if not np.allclose(kv_cache_output[layer_idx]["key_states"], kv_cache_output_golden[layer_idx]["key_states"][0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
            kv_cache_match = False
            break

    if kv_cache_match and np.allclose(att_output, att_output_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("att_output:", att_output[0, :10])
        print("golden att_output:", att_output_golden[0][0, 0, :10])


def test_action_forward():
    print("TEST test_action_forward")
    test_attention_2d_mask = np.ones((51, 867), dtype=np.bool)
    test_position_ids = np.arange(51, dtype=np.int64)
    test_suffix_embs = np.random.randn(51, 1024).astype(np.float32)

    att_output = paligemma_action_forward(
        attention_mask=test_attention_2d_mask,
        position_ids=test_position_ids,
        input_embeddings=test_suffix_embs,
        past_key_values=test_past_key_values,
    )

    att_output_golden, _ = policy.model.paligemma_with_expert.forward(
        attention_mask=torch.from_numpy(test_attention_2d_mask[None, ...]).to("cuda"),
        position_ids=torch.from_numpy(test_position_ids[None, ...]).to("cuda"),
        past_key_values=test_past_key_values_torch,
        inputs_embeds=[None, torch.from_numpy(test_suffix_embs[None, ...]).to("cuda")],
        use_cache=True,
        fill_kv_cache=False,
    )

    if np.allclose(att_output, att_output_golden[1].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("att_output:", att_output[0, :10])
        print("golden att_output:", att_output_golden[1][0, 0, :10])


def test_denoise_step():
    print("TEST test_denoise_step")
    test_state = np.random.randn(32,).astype(np.float32)
    test_prefix_pad_masks = np.ones((816,), dtype=np.bool)
    test_x_t = np.random.randn(50, 32).astype(np.float32)
    test_timestep = 0.5

    v_t = denoise_step(
        test_state,
        test_prefix_pad_masks,
        test_past_key_values,
        test_x_t,
        test_timestep,
    )
    v_t_golden = policy.model.denoise_step(
        torch.from_numpy(test_state[None, ...]).to("cuda"),
        torch.from_numpy(test_prefix_pad_masks[None, ...]).to("cuda"),
        test_past_key_values_torch,
        torch.from_numpy(test_x_t[None, ...]).to("cuda"),
        torch.tensor([test_timestep]).to("cuda"),
    )
    if np.allclose(v_t, v_t_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("v_t:", v_t[0, :10])
        print("golden v_t:", v_t_golden[0, 0, :10])


def test_sample_actions():
    print("TEST test_sample_actions")
    test_images = np.random.randn(3, 3, 224, 224).astype(np.float32)
    test_img_masks = np.ones((3,), dtype=np.bool)
    test_lang_tokens = np.random.randint(0, 100, (48,)).astype(np.int32)
    test_lang_masks = np.ones((48,), dtype=np.bool)
    test_state = np.random.randn(32,).astype(np.float32)
    test_noise = sample_noise((50, 32))

    actions = sample_actions(
        images=test_images,
        img_masks=test_img_masks,
        lang_tokens=test_lang_tokens,
        lang_masks=test_lang_masks,
        state=test_state,
        noise=test_noise.copy(),
    )

    actions_golden = policy.model.sample_actions(
        images=[torch.from_numpy(test_images[img_idx, ...][None, ...]).to("cuda") for img_idx in range(test_images.shape[0])],
        img_masks=[torch.tensor([1 if mask else 0], dtype=torch.bool).to("cuda") for mask in test_img_masks],
        lang_tokens=torch.from_numpy(test_lang_tokens[None, ...]).to("cuda"),
        lang_masks=torch.from_numpy(test_lang_masks[None, ...]).to("cuda"),
        state=torch.from_numpy(test_state[None, ...]).to("cuda"),
        noise=torch.from_numpy(test_noise[None, ...]).to("cuda"),
    )

    if np.allclose(actions, actions_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("actions:", actions[0, :10])
        print("golden actions:", actions_golden[0, 0, :10])


def test_select_action():
    print("TEST test_select_action")

    test_images = np.random.randn(3, 3, 224, 224).astype(np.float32)
    test_state = np.random.randn(12,).astype(np.float32)
    test_task = ["Just try to do something useful."]
    test_noise = sample_noise((50, 32))

    observation = {
        "observation.images.top": torch.from_numpy(test_images[0][None, ...]).cuda(),
        "observation.images.left_wrist": torch.from_numpy(test_images[1][None, ...]).cuda(),
        "observation.images.right_wrist": torch.from_numpy(test_images[2][None, ...]).cuda(),
        "observation.state": torch.from_numpy(test_state[None, ...]).cuda(),
        "task": test_task,
    }

    actions = select_action(
        images=test_images.copy(),
        state=test_state.copy(),
        task=test_task.copy(),
        noise=test_noise.copy(),
    )

    actions_golden = policy.select_action(
        observation,
        noise=torch.from_numpy(test_noise[None, ...]).to("cuda"),
    )

    # 0.1 as error tolerance somehow is not sufficient, might be a bug
    if np.allclose(actions, actions_golden.float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("actions:", actions[:3, :10])
        print("golden actions:", actions_golden[:3, :10])


test_rms_norm()
test_sample_noise()

tset_create_sinusoidal_pos_embedding()
test_embed_prefix()
test_embed_suffix()

test_make_att_2d_masks()
test_attention()
test_vlm_mlp_forward()
test_vlm_forward()
test_action_forward()

# overall test
test_denoise_step()
test_sample_actions()
test_select_action()
