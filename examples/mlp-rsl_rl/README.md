# Exporting Policy from RSL_RL Learning Framework

In RSL_RL learning framework, the trained policy is an `ActorCritic`  instance.

The model structure should look like the following. The exact dimension might be different depending on the specific task and environment the policy is trained in.

```python
>>> ppo_runner.alg.policy
ActorCritic(
  (actor): Sequential(
    (0): Linear(in_features=81, out_features=256, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=23, bias=True)
  )
  (critic): Sequential(
    (0): Linear(in_features=81, out_features=256, bias=True)
    (1): ELU(alpha=1.0)
    (2): Linear(in_features=256, out_features=128, bias=True)
    (3): ELU(alpha=1.0)
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): ELU(alpha=1.0)
    (6): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

During policy deployment, we only need to perform inference using the actor network. Hence, we only need to export the actor parameters.

In the evaluation script, insert the following line:

```python
torch.save(ppo_runner.alg.policy.actor, "./policy.pt")
```

This method will save the actor parameter as a `policy.pt` file. Move the file under this directory. Now we can follow the normal flow to convert this PyTorch model to run in baremetal C.


> ### Note
>
> We cannot use the checkpoints saved during training, which looks like `model_1000.pt`. These checkpoints only contain weights but not the model definition.
> 
> Some learning framework might export the policy as a PyTorch JIT exported module, which also commonly are called `policy.pt`. However, this is an instance of `RecursiveScriptModule` that does not work with our TracedModule, so this also cannot be used.

