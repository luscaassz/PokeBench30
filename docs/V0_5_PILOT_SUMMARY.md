# PokéBench30 v0.5.x — Pilot and Engineering Validation Summary

The v0.5.x experiments are retained as methodological pilot studies. They
motivated the decision to freeze the RL method before the main
cross-generation benchmark.

## Representation

UBR 1.2 added battle-context information to UBR 1.1, but the multi-seed
10k/50k studies did not show a systematic external win-rate improvement.

This supports using the richer universal representation for conceptual
coverage without claiming that it is an optimized representation.

## Training budget

Increasing direct TypeAware training from tens of thousands to hundreds of
thousands of PPO steps produced some improvement but also clear diminishing
returns.

Training reward and controlled external performance did not track each other
reliably.

## Curriculum

At 300k, Random -> TypeAware produced the best mean TypeAware result among the
tested schedules, but the gain over direct TypeAware training was modest.

Opponent sequence therefore affected learning enough that keeping a scripted
training opponent in the main generation comparison would create an avoidable
confound.

This motivates self-play in v0.6.

## Policy architecture

At 300k under Random -> TypeAware:

```text
                         TypeAware final mean
MLP baseline                  0.1367
history LSTM                  0.0700
large MLP                     0.0600
history MLP                   0.0367
```

The tested increases in feed-forward capacity and short four-decision temporal
context did not improve the primary benchmark.

## RL algorithm

Under the v0.5.6 algorithm study:

```text
                         Random   MaxDamage   TypeAware
MaskablePPO              0.8267     0.2867      0.1367
Masked Double DQN        0.4200     0.1167      0.0133
Masked DQN               0.3333     0.1033      0.0200
```

Under those settings, PPO clearly outperformed both value-based alternatives.

This does not establish that PPO is universally optimal for Pokémon. It is
sufficient evidence for the benchmark-design decision:

> use one established PPO protocol and stop treating RL selection as the
> central contribution.

## v0.6 consequence

The main benchmark therefore freezes:

```text
PPO
action masking
self-play
sparse terminal reward
Wang learning-rate schedule / PPO coefficients
UBR 1.2
common-core 10-action interface
```

and varies:

```text
Generation I / IV / IX
```

The pilot results remain useful as supplementary material showing how the final
experimental protocol was selected.
