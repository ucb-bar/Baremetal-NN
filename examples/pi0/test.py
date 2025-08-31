import numpy as np
import torch
from lerobot.policies.pi0.paligemma_with_expert import PaliGemmaWithExpertModel, PaliGemmaWithExpertConfig

from model import (
    eager_attention_forward,
    paligemma_with_expert_forward,
    gemma_mlp_forward,
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
    if np.allclose(out_emb, out_emb_gold.float().detach().cpu().numpy(), atol=0.5, rtol=0.5):
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
    if np.allclose(att_output, att_output_golden[0].float().detach().cpu().numpy(), atol=0.5, rtol=0.5):
        print(" ✔ TEST PASSED")
    else:
        print(" ✘ TEST FAILED")
        print("att_output:", att_output[0, 0, :10])
        print("golden att_output:", att_output_golden[0][0, 0, :10])


test_attention()
test_vlm_mlp_forward()
test_vlm_forward()
