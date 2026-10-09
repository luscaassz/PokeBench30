# PokéBench30 v0.5.3 — Training Budget Study

A v0.5.3 mantém as representações e o algoritmo da v0.5.2 e muda apenas a
pergunta experimental.

Na v0.5.2, com cinco seeds e 10k timesteps, obtivemos:

```text
                    Random      TypeAware
UBR 1.1             0.7033        0.0600
UBR 1.2             0.7033        0.0600
```

A v0.5.3 pergunta se a UBR 1.2 precisa de mais experiência para aproveitar o
battle context adicional.

## Representações

```text
UBR 1.1: 472 observações, 10 ações
UBR 1.2: 518 observações, 10 ações
```

Nenhuma feature ou ação foi alterada nesta versão.

Com o default `n_steps=2048`, os budgets são valores solicitados ao PPO.
O Stable-Baselines3 encerra após rollouts completos, então normalmente:

```text
10.000 solicitados → 10.240 executados
50.000 solicitados → 51.200 executados
```

O `metadata.json` e as tabelas registram o **budget solicitado**, enquanto os
logs do PPO mostram o total efetivamente coletado.

## Estudo padrão

```text
Budgets:
10k
50k

Representações:
UBR 1.1
UBR 1.2

Seeds:
42
123
456
789
2026

Treino:
Gen IX
MaskablePPO
vs TypeAware

Avaliação:
vs Random
vs TypeAware
3 fixed pairs
AB/BA mirrored
10 repeats por leg
```

## Novos arquivos

```text
run_budget_study.py
pokebench30/rl/budget_study.py
scripts/check_budget_study_setup.py
docs/TRAINING_BUDGET_v0.5.3.md
docs/V0_5_2_RESULTS.md
```

## Instalação

```powershell
python -m pip install -r requirements.txt
```

Prepare o fixed-team pool, se necessário:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

## Reutilizar seu experimento v0.5.2

Essa é a opção recomendada, pois você já executou os 10 modelos de 10k.

No seu caso, o experimento anterior foi:

```text
PokeBench30_v0.5.2\multiseed_results\multiseed_gen9_20260929_200322
```

Primeiro valide a compatibilidade:

```powershell
python scripts\check_budget_study_setup.py --baseline-experiment-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.2\multiseed_results\multiseed_gen9_20260929_200322"
```

Depois execute:

```powershell
python run_budget_study.py --baseline-experiment-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.2\multiseed_results\multiseed_gen9_20260929_200322"
```

Isso importa os resultados de 10k e executa somente os dez modelos novos de
50k.

## Carga nova de trabalho

Com baseline 10k importada:

```text
10 modelos novos
500.000 timesteps novos no total

10 modelos
× 2 adversários
× 60 batalhas
=
1.200 novas batalhas de avaliação
```

## Smoke test

Antes do estudo completo, você pode testar a automação sem importar baseline:

```powershell
python run_budget_study.py --budgets 1024 2048 --seeds 42 123 --repeats 1 --n-steps 256 --batch-size 64
```

Não use esse smoke como resultado científico.

## Estrutura de saída

```text
budget_results/
└── budget_gen9_YYYYMMDD_HHMMSS/
    ├── study_manifest.json
    ├── imported_baseline/
    ├── budget_50000/
    │   ├── models/
    │   └── evaluations/
    └── aggregate/
        ├── training_summary.csv
        ├── per_model_budget_results.csv
        ├── aggregate_summary.csv
        ├── representation_paired_differences.csv
        ├── representation_comparison_by_budget.csv
        ├── budget_gain_by_seed.csv
        ├── budget_comparison.csv
        ├── budget_representation_interaction.csv
        └── figures/
```

## Arquivo principal para a pergunta da v0.5.3

```text
budget_representation_interaction.csv
```

Ele calcula descritivamente:

```text
ganho UBR 1.2 de 10k → 50k
-
ganho UBR 1.1 de 10k → 50k
```

Se for positivo, a UBR 1.2 melhorou mais com orçamento adicional; se for
negativo, a UBR 1.1 melhorou mais.

Com cinco seeds, isso deve ser interpretado como evidência descritiva, não como
teste definitivo de equivalência ou superioridade.

## Resume

O estudo é retomável.

Quando o programa informar:

```text
[INFO] Budget-study directory:
...\budget_results\budget_gen9_YYYYMMDD_HHMMSS
```

guarde esse caminho.

Se houver interrupção:

```powershell
python run_budget_study.py --study-dir "budget_results\budget_gen9_YYYYMMDD_HHMMSS" --baseline-experiment-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.2\multiseed_results\multiseed_gen9_20260929_200322"
```

Use sempre os mesmos parâmetros ao retomar.

## Continuidade do projeto

```text
v0.5     RL baseline
v0.5.1   battle-context representation
v0.5.2   5-seed UBR ablation at 10k
v0.5.3   10k vs 50k training-budget study
v0.6     cross-generation transfer
```
