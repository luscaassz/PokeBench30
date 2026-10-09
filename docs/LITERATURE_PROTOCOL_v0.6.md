# Literature protocol adopted in PokéBench30 v0.6

PokéBench30 v0.6 stops treating reinforcement-learning design as the central
research question. The RL agent is now a **frozen experimental instrument** for
comparing Pokémon generations.

The principal methodological reference is:

> Jett Wang. *Winning at Pokémon Random Battles Using Reinforcement Learning*.
> MIT MEng thesis, 2024.

A secondary scale reference is:

> Leonardo de Lellis Rossi et al. *I Choose You, Reinforcement Learning!
> Trained RL Agents For Pokémon Battles*. SBGames 2024.

## What is adopted from Wang

The v0.6 training protocol adopts the following ideas directly:

- PPO actor-critic training;
- invalid-action masking;
- self-play rather than a hand-authored training opponent;
- sparse terminal reward:
  - `+1` win;
  - `-1` loss;
  - `0` otherwise;
- learning-rate annealing:

```text
lr(x) = 10^-4.23 / (8x + 1)^1.5
```

where `x` is training progress from 0 to 1;

- published PPO optimization values:

```text
n_epochs          7
gamma             0.9999
gae_lambda        0.754
clip_range        0.0829
clip_range_vf     0.0184
entropy_coef      0.0588
value_coef        0.4375
max_grad_norm     0.5430
batch_size        1024
hidden_dim        256
```

- architectural shape:
  - shared three-layer ReLU trunk with hidden dimension 256;
  - separate two-layer actor and critic heads.

## What is intentionally not reproduced exactly

This is **not** presented as a reproduction of Wang's agent.

Wang's original system used:

- a 3725-dimensional generation-specific state;
- learned embeddings for species, ability, item and move identities;
- 78 workers / 39 parallel games;
- both players' trajectories added to training;
- 150 million training steps;
- MCTS as an inference-time policy-improvement operator.

PokéBench30 changes these components because its scientific goal is different.

### Universal representation

PokéBench30 keeps **UBR 1.2 (518 features)** so the same interface can be used
in Generations I, IV and IX.

### Portable two-player PettingZoo rollout

The original `78 × 512` worker layout cannot be reproduced on the target
single-machine setup without turning distributed systems engineering into the
research contribution.

v0.6 uses poke-env's two-player PettingZoo interface with one battle-environment
copy:

```text
2 player perspectives
× n_steps 2048
= 4096 transitions per PPO rollout

batch_size = 1024
```

Both perspectives are controlled by the same policy and both trajectories enter
PPO, preserving the core self-play property of Wang's setup while using far
fewer parallel workers.

### Shared-policy self-play

There is no scripted training opponent. Both players in every training battle
use the same current PPO policy.

This is also consistent with poke-env's official self-play pattern, where the
two-agent PettingZoo environment is converted to a Stable-Baselines3 vector
environment and optimized with one shared policy.

### No MCTS

MCTS is excluded from v0.6.0.

The comparison is intended to measure how the **same learned policy system**
behaves across generations, not to build the strongest possible Pokémon bot.
MCTS would introduce a second planning algorithm, unknown-information sampling,
and generation-dependent simulation concerns.

## Why 900k training steps

Wang's 150M-step run is outside the practical scope of a
three-generation × five-seed benchmark.

Rossi et al. explicitly evaluate DRL agents at 300k and 900k steps. v0.6 uses
900k nominal steps per generation/seed as a practical literature-grounded
budget.

This does **not** imply that 900k is equivalent to Wang's convergence scale.

## Common-core action space

PokéBench30 v0.6.0 keeps the universal 10-action space:

```text
4 move slots
6 own-team slots
```

Generation-specific gimmicks such as Terastallization remain excluded.

This means v0.6.0 is explicitly a **common-core cross-generation benchmark**:
the action semantics are identical across Generations I, IV and IX.

A later full-mechanics extension may introduce a larger universal action space,
but that is deliberately outside the first cross-generation experiment.
