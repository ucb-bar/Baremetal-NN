import numpy as np
import torch
from lerobot.policies.pi0.paligemma_with_expert import PaliGemmaWithExpertModel, PaliGemmaWithExpertConfig
from lerobot.policies.pi0.modeling_pi0 import create_sinusoidal_pos_embedding as create_sinusoidal_pos_embedding_golden

from model import (
    eager_attention_forward,
    paligemma_with_expert_forward,
    gemma_mlp_forward,
    create_sinusoidal_pos_embedding,
    embed_suffix,
    sample_actions,
    policy,
)


def test_attention():
    test_attention_mask = np.tril(np.ones((816,), dtype=np.bool))[None, :, :]
    test_batch_size = 1
    test_head_dim = 256
    test_query_states = np.random.randn(1, 816, 8, 256).astype(np.float32)
    test_key_states = np.random.randn(1, 816, 1, 256).astype(np.float32)
    test_value_states = np.random.randn(1, 816, 1, 256).astype(np.float32)

    # our implementation does not support batching
    att_output = eager_attention_forward(
        test_attention_mask[0, ...],
        test_head_dim,
        test_query_states[0, ...],
        test_key_states[0, ...],
        test_value_states[0, ...],
    )

    model = PaliGemmaWithExpertModel(PaliGemmaWithExpertConfig())

    att_output_golden = model.eager_attention_forward(
        torch.from_numpy(test_attention_mask),
        test_batch_size,
        test_head_dim,
        torch.from_numpy(test_query_states),
        torch.from_numpy(test_key_states),
        torch.from_numpy(test_value_states)
    )
    print("TEST test_attention")
    if np.allclose(att_output, att_output_golden, atol=1e-4, rtol=1e-4):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test att_output:", att_output[0, 0, :10])
        print("golden att_output:", att_output_golden[0, 0, :10])


def test_vlm_mlp_forward():
    # Test MLP forward
    test_input_emb = np.random.randn(1, 816, 2048).astype(np.float32)

    model = policy.model.paligemma_with_expert.paligemma.language_model
    # our implementation does not support batching
    out_emb = gemma_mlp_forward(model.layers[0].mlp, test_input_emb[0, ...])

    out_emb_gold = policy.model.paligemma_with_expert.paligemma.language_model.layers[0].mlp(
        torch.from_numpy(test_input_emb).to("cuda")
    )

    print("TEST test_vlm_mlp_forward")
    if np.allclose(out_emb, out_emb_gold.float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("test out_emb:", out_emb[0, 0, :10])
        print("golden out_emb:", out_emb_gold[0, 0, :10])


def test_vlm_forward():
    test_attention_2d_mask = np.tril(np.ones((816,), dtype=np.bool))[None, :, :]
    test_position_ids = np.arange(816, dtype=np.int64)[None, :]
    test_prefix_embs = np.random.randn(1, 816, 2048).astype(np.float32)

    att_output, _ = paligemma_with_expert_forward(
        attention_mask=test_attention_2d_mask[0, ...],
        position_ids=test_position_ids[0, ...],
        input_embeds=test_prefix_embs[0, ...],
    )

    att_output_golden, _ = policy.model.paligemma_with_expert.forward(
        attention_mask=torch.from_numpy(test_attention_2d_mask).to("cuda"),
        position_ids=torch.from_numpy(test_position_ids).to("cuda"),
        past_key_values=None,
        inputs_embeds=[torch.from_numpy(test_prefix_embs).to("cuda"), None],
        use_cache=True,
        fill_kv_cache=True,
    )

    print("TEST test_vlm_forward")
    if np.allclose(att_output, att_output_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("att_output:", att_output[0, 0, :10])
        print("golden att_output:", att_output_golden[0][0, 0, :10])


def tset_create_sinusoidal_pos_embedding():
    test_timestep = np.array([0.5], dtype=np.float32)
    test_proj_width = 2048

    time_emb = create_sinusoidal_pos_embedding(
        test_timestep,
        test_proj_width,
        min_period=4e-3,
        max_period=4.0,
    )
    time_emb_golden = create_sinusoidal_pos_embedding_golden(
        torch.from_numpy(test_timestep).to("cuda"),
        test_proj_width,
        min_period=4e-3,
        max_period=4.0,
        device=torch.device("cuda"),
    )

    print("TEST test_create_sinusoidal_pos_embedding")
    if np.allclose(time_emb, time_emb_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("time_emb:", time_emb[0, 0, :10])
        print("golden time_emb:", time_emb_golden[0][0, 0, :10])


def test_embed_suffix():
    test_state = np.random.randn(1, 32).astype(np.float32)
    test_noisy_actions = np.random.randn(1, 50, 32).astype(np.float32)
    test_timestep = np.array([0.5], dtype=np.float32)

    suffix_embs, suffix_pad_masks, suffix_att_masks = embed_suffix(
        test_state, test_noisy_actions, test_timestep
    )
    suffix_embs_golden, suffix_pad_masks_golden, suffix_att_masks_golden = policy.model.embed_suffix(
        torch.from_numpy(test_state).to("cuda"),
        torch.from_numpy(test_noisy_actions).to("cuda"),
        torch.from_numpy(test_timestep).to("cuda"),
    )
    print("TEST test_embed_suffix")
    if np.allclose(suffix_embs, suffix_embs_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("suffix_embs:", suffix_embs[0, 0, :10])
        print("golden suffix_embs:", suffix_embs_golden[0, 0, :10])


def test_sample_actions():
    test_images = [np.random.randn(1, 3, 224, 224).astype(np.float32)] * 3
    test_img_masks = [np.ones((1,), dtype=np.bool)] * 3
    test_lang_tokens = np.random.randint(0, 100, (1, 48)).astype(np.int32)
    test_lang_masks = np.ones((1, 48), dtype=np.bool)
    test_state = np.random.randn(1, 32).astype(np.float32)

    actions = sample_actions(
        images=test_images,
        img_masks=test_img_masks,
        lang_tokens=test_lang_tokens,
        lang_masks=test_lang_masks,
        state=test_state,
    )

    actions_golden = policy.model.sample_actions(
        images=[torch.from_numpy(img).to("cuda") for img in test_images],
        img_masks=[torch.from_numpy(img_mask).to("cuda") for img_mask in test_img_masks],
        lang_tokens=torch.from_numpy(test_lang_tokens).to("cuda"),
        lang_masks=torch.from_numpy(test_lang_masks).to("cuda"),
        state=torch.from_numpy(test_state).to("cuda"),
    )

    print("TEST test_sample_actions")
    if np.allclose(actions, actions_golden[0].float().detach().cpu().numpy(), atol=0.1, rtol=0.1):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("actions:", actions[0, 0, :10])
        print("golden actions:", actions_golden[0, 0, :10])


# test_attention()
tset_create_sinusoidal_pos_embedding()
test_embed_suffix()
# test_vlm_mlp_forward()
# test_vlm_forward()
# test_sample_actions()
