# PokéBench30 v0.5.1 — UBR 1.2 RL ablation

v0.5.1 is not a new RL algorithm. It is a controlled representation change.

```text
v0.5:
MaskablePPO + UBR 1.1 (472)

v0.5.1:
MaskablePPO + UBR 1.2 (518)
```

Keep the following fixed for the first comparison:

```text
generation = 9
opponent = type_aware
timesteps = 10000
seed = 42
learning_rate = 0.0003
n_steps = 2048
batch_size = 128
reward = unchanged
policy net = [256, 128]
```

## Required validation before training

1. Run `scripts/check_rl_setup.py`.
2. Run `scripts/check_representation.py`.
3. Generate the fixed-team pool.
4. Run a six-battle scripted smoke test with universal-state logging.
5. Run `scripts/validate_universal_states.py`.
6. Require all structural checks to be zero.

## Training

```powershell
python train_rl.py --generation 9 --opponent type_aware --timesteps 10000 --seed 42
```

Old v0.5 UBR 1.1 models are intentionally incompatible with v0.5.1 and should
fail metadata validation.

## Controlled evaluation

Against RandomAgent:

```powershell
python run_rl_controlled.py --model "MODEL" --generation 9 --opponent random --repeats 10
```

Against TypeAwareAgent:

```powershell
python run_rl_controlled.py --model "MODEL" --generation 9 --opponent type_aware --repeats 10
```

The primary comparison is the normalized mirrored score under the same fixed
team pool used for the UBR 1.1 reference.
