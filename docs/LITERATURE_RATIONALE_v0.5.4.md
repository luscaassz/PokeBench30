# Literature rationale for v0.5.4

## Wang (MIT, 2024)

**Winning at Pokémon Random Battles Using Reinforcement Learning** — Jett Wang.

Relevant design observations for PokéBench30:

- models Pokémon Random Battles as a partially observable problem;
- trains an actor-critic network with PPO using self-play;
- masks invalid actions;
- validates repeatedly against a heuristic opponent rather than relying only on PPO training loss/reward;
- uses a much larger state representation and training scale than the current PokéBench baseline;
- reports a large benefit from learning-rate annealing;
- uses MCTS at inference as a policy-improvement mechanism.

PokéBench30 v0.5.4 adopts the **checkpoint external validation principle**, but intentionally does not adopt self-play, LR annealing or MCTS yet, because the current experiment isolates curriculum.

## Rossi et al. (SBGames 2024)

**I Choose You, Reinforcement Learning! Trained RL Agents For Pokémon Battles**.

Relevant observations:

- compares multiple tabular and deep RL methods;
- trains DRL agents at 300k and 900k steps;
- uses RandomPlayer and MaxDamagePlayer as training/evaluation opponents;
- uses a dense reward based on HP, remaining Pokémon, status, fainting and battle outcome.

This supports 300k as a reasonable next experimental scale for PokéBench30 and motivates including Random and MaxDamage in the curriculum.

## Ivison (2021)

**PokéRL: Pokémon + Reinforcement Learning**.

The reported experiments compare agents trained against Random and MaxDamage-style opponents. The discussion raises the possibility that starting from a more diverse/simple opponent may yield better generalization than directly fitting to a stronger deterministic opponent.

This directly motivates the `random_to_type_aware` and `progressive` arms.

Source: https://ivison.id.au/2021/08/02/pokerl.html

## Vivek — Pokémon Showdown with Reinforcement Learning

The article demonstrates a poke-env RL setup with a compact observation and reward based on fainting, HP and victory, conceptually close to the reward already used by PokéBench30.

This is one reason v0.5.4 leaves reward shaping unchanged and first tests the training regime.

Source: https://medium.com/@kevalvivek4/pokémon-showdown-with-reinforcement-learning-f5aad9a93cef

## Rossi et al. open-source implementation

Repository:

https://github.com/leolellisr/poke_RL

It is the implementation associated with the SBGames work above, so it should not be treated as independent experimental evidence from that paper.
