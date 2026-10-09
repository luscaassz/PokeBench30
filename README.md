# PokéBench30 v0.5.5 — Policy Architecture Study

A v0.5.5 investiga se o gargalo observado nas versões anteriores está na
**arquitetura da policy**.

O resultado da v0.5.4 mostrou que `Random -> TypeAware` foi o curriculum com a
maior média final contra TypeAware:

```text
Direct TypeAware        0.110
Random -> TypeAware     0.137
Progressive             0.057
Random only             0.077
```

Por isso, a v0.5.5 congela esse curriculum e muda apenas a policy.

## Condições fixas

```text
Generation       = IX
UBR              = 1.2
UBR obs          = 518
Action space     = 10
Algorithm        = MaskablePPO
Reward           = igual à v0.5-v0.5.4
Learning rate    = 3e-4 constante
Seeds            = 42, 123, 456, 789, 2026
Budget           = 300k nominal / trajetória

Curriculum:
0-100k           Random
100-300k         TypeAware
```

## Policies comparadas

```text
mlp_baseline
518 -> [256, 128]

mlp_large
518 -> [512, 256, 128]

history_mlp
4 × 518 -> [512, 256, 128]

history_lstm
4 × 518 -> LSTM(256) -> [128]
```

`history_mlp` é importante porque separa o efeito de **ter histórico** do efeito
de **usar LSTM**.

A janela temporal tem quatro decisões e usa zero-padding no início da batalha.

## Por que não usar RecurrentPPO puro?

O benchmark depende de action masking estrito. A versão atual do
`MaskablePPO` usada pelo projeto não oferece policy recorrente nativa.

Trocar `MaskablePPO` por `RecurrentPPO` mudaria também o tratamento de ações
ilegais e tornaria a comparação injusta.

Por isso, `history_lstm` usa uma LSTM como **feature extractor temporal de
janela fixa**, mantendo exatamente o mesmo MaskablePPO e as mesmas máscaras.

## Instalação

```powershell
python -m pip install -r requirements.txt
```

Se necessário, prepare novamente o pool:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

## Reutilizar sua v0.5.4

Pelos seus logs, o estudo completo anterior foi:

```text
PokeBench30_v0.5.4\curriculum_results\curriculum_gen9_20260930_000136
```

Na nova pasta, valide primeiro:

```powershell
python scripts\check_policy_study_setup.py --baseline-curriculum-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.4\curriculum_results\curriculum_gen9_20260930_000136"
```

Se aparecer tudo `[OK]`, o baseline pode ser importado.

## Smoke test

O smoke é independente e treina todas as quatro policies em escala pequena:

```powershell
$env:MPLBACKEND="Agg"
python run_policy_study.py --smoke
```

Configuração do smoke:

```text
1 seed
4 architectures
6144 nominal timesteps
checkpoints em 2048 / 4096 / 6144
Random 2048 -> TypeAware 4096
1 repeat por mirror leg
```

Os resultados do smoke não são científicos; ele serve para validar:

- temporal stack;
- LSTM extractor;
- model save/load;
- architecture-aware inference;
- action masking;
- checkpoints;
- controlled evaluation.

## Experimento completo recomendado

Depois do smoke:

```powershell
$env:MPLBACKEND="Agg"

python run_policy_study.py --baseline-curriculum-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.4\curriculum_results\curriculum_gen9_20260930_000136"
```

Como o `mlp_baseline` é importado da v0.5.4, serão treinadas apenas:

```text
mlp_large
history_mlp
history_lstm

3 architectures
× 5 seeds
× 300k
=
4.5M novos timesteps nominais
```

## Avaliação

Cada trajectory é observada em:

```text
50k
100k
150k
200k
250k
300k
```

contra:

```text
Random
MaxDamage
TypeAware
```

Os checkpoints intermediários são diagnósticos.
O checkpoint final usa 60 batalhas por seed/adversário.

## Resultados

```text
policy_results/
└── policy_gen9_.../
    ├── study_manifest.json
    ├── imported_mlp_baseline/
    ├── runs/
    │   ├── mlp_large/
    │   ├── history_mlp/
    │   └── history_lstm/
    └── aggregate/
        ├── learning_curve.csv
        ├── aggregate_learning_curve.csv
        ├── training_progress.csv
        ├── final_summary.csv
        ├── paired_architecture_differences.csv
        ├── architecture_comparison.csv
        └── figures/
```

O arquivo principal é:

```text
architecture_comparison.csv
```

Ele compara cada arquitetura com `mlp_baseline` por seed.

## Resume

Ao iniciar, guarde o diretório impresso:

```text
policy_results\policy_gen9_YYYYMMDD_HHMMSS
```

Se o processo parar:

```powershell
python run_policy_study.py --study-dir "policy_results\policy_gen9_YYYYMMDD_HHMMSS" --baseline-curriculum-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.4\curriculum_results\curriculum_gen9_20260930_000136"
```

Use os mesmos parâmetros.

## Sequência experimental

```text
v0.5.2  representação
v0.5.3  training budget
v0.5.4  curriculum
v0.5.5  policy architecture
v0.6    cross-generation transfer
```

Mais detalhes: `docs/POLICY_ARCHITECTURE_v0.5.5.md`.
