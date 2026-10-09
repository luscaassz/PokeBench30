# PokéBench30 v0.5.2 — Multi-seed Representation Ablation

v0.5.2 automates the next experiment in PokéBench30: compare the two already
validated universal battle representations across several independent
MaskablePPO runs.

```text
UBR 1.1: 472 observations + 10 actions
UBR 1.2: 518 observations + 10 actions
```

Both representations now live in the **same codebase**, which avoids comparing
models trained by two different project snapshots.

## What changed

The RL command now accepts either representation:

```powershell
python train_rl.py --generation 9 --ubr ubr-1.1 --opponent type_aware --timesteps 10000 --seed 42
python train_rl.py --generation 9 --ubr ubr-1.2 --opponent type_aware --timesteps 10000 --seed 42
```

`RLPolicyAgent` reads `metadata.json` and automatically uses the correct encoder
when a saved model is evaluated.

The main new entry point is:

```text
run_multiseed.py
```

It trains both UBR profiles, evaluates every model with the same fixed/mirrored
benchmark, and creates aggregate CSVs and plots.

---

## 1. Install and check

```powershell
python -m pip install -r requirements.txt
python scripts\check_rl_setup.py
```

The checker should confirm:

```text
UBR 1.1: 472 observations, 10 actions
UBR 1.2: 518 observations, 10 actions
UBR 1.2 preserves the exact 472-feature UBR 1.1 prefix
```

You can also test either encoder offline:

```powershell
python scripts\check_representation.py --ubr ubr-1.1
python scripts\check_representation.py --ubr ubr-1.2
```

---

## 2. Prepare the fixed benchmark once

If `teams\fixed\manifest.json` does not exist:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

Then:

```powershell
python scripts\check_multiseed_setup.py
```

Keep the local Pokémon Showdown server running during training/evaluation, as
in the previous versions.

---

## 3. Recommended small smoke test

Before launching the complete experiment:

```powershell
python run_multiseed.py --seeds 42 123 --timesteps 1024 --repeats 1 --n-steps 256 --batch-size 64
```

This trains four small models:

```text
UBR 1.1 seed 42
UBR 1.1 seed 123
UBR 1.2 seed 42
UBR 1.2 seed 123
```

and evaluates them against Random and TypeAware with only one repeat per mirror
leg. This is a pipeline test only.

---

## 4. Official v0.5.2 experiment

After the smoke test succeeds, the default command is simply:

```powershell
python run_multiseed.py
```

Defaults:

```text
Generation              Gen IX
UBRs                    1.1 and 1.2
Seeds                   42, 123, 456, 789, 2026
Training opponent       TypeAwareAgent
Timesteps/model         10,000
PPO n_steps             2,048
PPO batch size          128
Evaluation opponents    RandomAgent, TypeAwareAgent
Repeats/mirror leg      10
Pairs                   pair01, pair02, pair03
```

This creates:

```text
10 models
20 controlled evaluation runs
1,200 controlled evaluation battles
```

The experiment directory is printed immediately when the command starts.

---

## 5. Resume an interrupted run

Suppose the runner printed:

```text
multiseed_results\multiseed_gen9_20260928_160000
```

Rerun:

```powershell
python run_multiseed.py --experiment-dir "multiseed_results\multiseed_gen9_20260928_160000"
```

The configuration is checked. Completed models and evaluations are reused;
missing work continues.

---

## 6. Main outputs

Inside `aggregate\`:

```text
training_summary.csv
```

One row/model with episode count, mean training reward, final-window reward,
and episode length.

```text
per_model_results.csv
```

One row for each seed × UBR × evaluation opponent.

```text
aggregate_summary.csv
```

Mean and sample standard deviation of mirrored score across seeds.

```text
paired_differences.csv
```

For each seed/opponent:

```text
UBR 1.2 mirror score - UBR 1.1 mirror score
```

```text
representation_comparison.csv
```

Compact UBR-level comparison based on the matched seed labels.

The `figures\` folder plots the seed-wise UBR 1.1 → UBR 1.2 change separately
for each opponent.

---

## 7. Scientific interpretation

The experiment unit for representation stability is the **trained model/seed**,
not an individual battle. The 60 battles per evaluation improve measurement of
one model's performance on the three fixed pairs, but they do not create 60
independent training replicates.

Also, the same PPO seed does not imply identical Pokémon Showdown battle RNG
between UBR 1.1 and UBR 1.2. Once the representation changes, trajectories
change. v0.5.2 therefore uses matched seed labels for organization and paired
summaries, but does not claim a strict common-random-numbers experiment.

---

## 8. Current research sequence

```text
v0.3   fixed teams + mirrored benchmark
v0.4.1 UBR 1.1 structurally validated
v0.5   MaskablePPO + UBR 1.1 baseline
v0.5.1 UBR 1.2 battle-context ablation, seed 42
v0.5.2 multi-seed UBR 1.1 vs UBR 1.2
v0.6   cross-generation transfer (after representation selection)
```

See `docs\MULTISEED_v0.5.2.md` for the experiment design and
`CHANGELOG_v0.5.2.md` for implementation details.
