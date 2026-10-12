import numpy as np
import torch

from model import select_action, sample_noise, policy


def main():

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
        print(" ✔ Output matches")
    else:
        print(" ✘ Output does not match")

    print("actions:", actions[:3, :10])
    print("golden actions:", actions_golden[:3, :10])


if __name__ == "__main__":
    main()
