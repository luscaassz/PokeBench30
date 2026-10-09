# PokéBench30 v0.5.6 — RL Algorithm Study

A v0.5.6 testa a próxima hipótese do projeto:

> O gargalo atual está no algoritmo de aprendizado, e não na arquitetura da
> policy?

A v0.5.5 mostrou que a MLP original `[256,128]` continuou sendo a melhor das
arquiteturas testadas contra TypeAware:

```text
MLP baseline       13.67%
History LSTM        7.00%
MLP large           6.00%
History MLP         3.67%
```

Por isso a v0.5.6 congela a arquitetura original e compara três algoritmos.

## Algoritmos

```text
MaskablePPO
Masked DQN
Masked Double DQN
```

Todos usam:

```text
Generation        IX
UBR               1.2
Observation       518
Action space      10
Policy/Q network  [256, 128]
Reward            igual à v0.5-v0.5.5
Seeds             42, 123, 456, 789, 2026

Curriculum:
0-100k            Random
100-300k          TypeAware
```

## Action masking no DQN/DDQN

O DQN não pode simplesmente usar `argmax Q(s,a)` sobre as 10 ações, porque
algumas são ilegais em cada turno.

A v0.5.6 mascara ações ilegais em dois pontos:

```text
1. escolha epsilon-greedy da ação
2. bootstrap do próximo estado
```

No Double DQN:

```text
online network
    escolhe a melhor ação legal

target network
    avalia essa ação
```

Assim uma ação ilegal nunca entra no `max`/`argmax` do target.

## Mesmo orçamento efetivo de experiência

Nos estudos anteriores, cada bloco nominal de 50k do PPO contém na prática:

```text
ceil(50000 / 2048) × 2048
= 51200 environment steps
```

Por isso o modelo final do PPO recebe:

```text
6 × 51200
= 307200 environment interactions
```

A v0.5.6 dá exatamente **307200 interações** também ao DQN e ao Double DQN.

Os gráficos continuam usando os rótulos nominais:

```text
50k
100k
150k
200k
250k
300k
```

## Configuração DQN/DDQN

```text
learning rate           2.5e-4
gamma                   0.99
batch size              128
replay buffer           50,000
learning starts         5,000
train frequency         4
gradient steps          1
target update           2,000
epsilon                 1.00 -> 0.05
epsilon decay           primeiros 50% do treino
loss                    Huber
network                 [256, 128]
```

DQN e Double DQN usam exatamente os mesmos valores; a única diferença entre
eles é a regra do bootstrap target.

## Hotfixes incorporados

A v0.5.6 já inclui as duas correções encontradas durante a v0.5.5:

```text
valid_orders suplementa available_moves
```

para evitar o erro de movimentos legais como Earthquake não serem mapeados, e

```text
canonicalização do study_manifest
```

para permitir resume correto depois da serialização JSON.

## 1. Copie o fixed-team pool

A nova versão não distribui novamente os times gerados.

Dentro da pasta v0.5.6:

```powershell
Remove-Item ".\teams\fixed" -Recurse -Force -ErrorAction SilentlyContinue

Copy-Item `
  "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.5\teams\fixed" `
  ".\teams\fixed" `
  -Recurse
```

Confira:

```powershell
Test-Path ".\teams\fixed\manifest.json"
```

Deve retornar `True`.

## 2. Valide o setup e a baseline PPO

O seu estudo completo da v0.5.5 foi:

```text
PokeBench30_v0.5.5\policy_results\policy_gen9_20261002_194318
```

Rode:

```powershell
python scripts\check_algorithm_study_setup.py `
  --ppo-baseline-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.5\policy_results\policy_gen9_20261002_194318"
```

O checker valida:

- UBR e SHA;
- fixed-team pool;
- seeds;
- curriculum;
- checkpoints;
- adversários;
- protocolo final;
- compatibilidade da baseline PPO.

## 3. Smoke test

Antes do experimento real:

```powershell
$env:MPLBACKEND="Agg"

python run_algorithm_study.py --smoke
```

O smoke treina:

```text
1 seed
3 algorithms
6144 nominal/effective steps
Random 2048 -> TypeAware 4096
```

com avaliações em:

```text
2048
4096
6144
```

Os resultados do smoke não devem ser interpretados cientificamente.

## 4. Experimento completo recomendado

Depois do smoke:

```powershell
$env:MPLBACKEND="Agg"

python run_algorithm_study.py `
  --ppo-baseline-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.5\policy_results\policy_gen9_20261002_194318"
```

Como o PPO será importado, apenas dois novos algoritmos serão treinados:

```text
Masked DQN
Masked Double DQN
```

Carga nova:

```text
2 algorithms
× 5 seeds
× 307200 effective environment steps

= 10 novas trajetórias
= 3,072,000 novas interações
```

## Resume

Guarde o diretório impresso no início:

```text
algorithm_results\algorithm_gen9_YYYYMMDD_HHMMSS
```

Se parar:

```powershell
python run_algorithm_study.py `
  --study-dir ".\algorithm_results\algorithm_gen9_YYYYMMDD_HHMMSS" `
  --ppo-baseline-dir "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.5\policy_results\policy_gen9_20261002_194318"
```

Para DQN/DDQN, o projeto guarda também o replay buffer do checkpoint mais
recente. O runner se recusa a continuar uma trajetória incompleta caso esse
replay não esteja disponível, em vez de resetá-lo silenciosamente.

## Resultados

```text
algorithm_results/
└── algorithm_gen9_.../
    ├── study_manifest.json
    ├── imported_maskable_ppo/
    ├── runs/
    │   ├── masked_dqn/
    │   └── masked_double_dqn/
    └── aggregate/
        ├── learning_curve.csv
        ├── aggregate_learning_curve.csv
        ├── training_progress.csv
        ├── final_summary.csv
        ├── paired_algorithm_differences.csv
        ├── algorithm_comparison.csv
        └── figures/
```

O arquivo principal será:

```text
algorithm_comparison.csv
```

A comparação mais importante continuará sendo:

```text
vs TypeAware @ 300k
```

pareada pelas mesmas cinco seeds.

## Sequência experimental

```text
v0.5.2  representação
v0.5.3  training budget
v0.5.4  curriculum
v0.5.5  policy architecture
v0.5.6  RL algorithm
```

Detalhes metodológicos:
`docs/RL_ALGORITHM_v0.5.6.md`.
