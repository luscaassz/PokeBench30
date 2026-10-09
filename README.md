# PokéBench30 v0.6.0 — Cross-Generation Benchmark

A partir da v0.6, o projeto deixa de tratar **qual RL é melhor** como a
pergunta central.

As versões v0.5.x ficam como estudos piloto / validação de engenharia.

A pergunta principal agora é:

> **Como o mesmo agente de Reinforcement Learning aprende e se comporta nas
> Gerações I, IV e IX de Pokémon quando representação, algoritmo,
> hiperparâmetros, orçamento de treino e avaliação são mantidos constantes?**

## O agente oficial da v0.6

O benchmark congela uma configuração de PPO inspirada diretamente em:

**Jett Wang — _Winning at Pokémon Random Battles Using Reinforcement Learning_
(MIT, 2024).**

A escala prática de 900k steps também é compatível com uma das condições
experimentais de Rossi et al. (SBGames 2024).

A configuração oficial passa a ser:

```text
Algorithm          PPO + custom action masking
Representation     UBR 1.2
Observation dim    518
Action dim         10
Training           online self-play
Reward             +1 win / -1 loss / 0 otherwise

PPO:
n_epochs           7
gamma              0.9999
gae_lambda          0.754
clip_range          0.0829
clip_range_vf       0.0184
entropy_coef        0.0588
value_coef          0.4375
max_grad_norm       0.5430
batch_size          1024

Network:
shared trunk        3 × 256 ReLU
actor head          2 × 256
critic head         2 × 256
```

Learning rate:

```text
lr(x) = 10^-4.23 / (8x + 1)^1.5
```

com `x` indo de 0 a 1 durante o treinamento.

## O que significa "inspirado em Wang"

Não estamos afirmando reproduzir exatamente o agente da tese.

Wang usa, entre outras coisas:

```text
3725-dimensional Gen-IV state
categorical embeddings
78 workers
both-player trajectories
150M steps
MCTS at inference
```

PokéBench30 precisa usar **a mesma interface nas três gerações**.

Por isso v0.6 mantém:

```text
UBR 1.2
1 PettingZoo battle environment copy
both-player PPO trajectories
900k nominal policy transitions
sem MCTS
```

Detalhes: `docs/LITERATURE_PROTOCOL_v0.6.md`.

---

# Experimento principal

```text
Gen I
Gen IV
Gen IX

× 5 seeds

42
123
456
789
2026

× 900k nominal steps
```

Total:

```text
15 modelos
```

Com `n_steps=2048` e duas perspectivas de jogador no VecEnv, cada rollout
contém:

```text
2048 × 2 = 4096 transitions
```

Cada modelo deve terminar aproximadamente em:

```text
901120 policy transitions
```

Total esperado:

```text
13,516,800 policy transitions
```

Cada modelo é treinado **do zero dentro de sua própria geração**.

v0.6.0 não faz transferência de pesos Gen I → IV → IX.

---

# Self-play

Durante o treino:

```text
same PPO policy
   ↙       ↘
Player 1  Player 2
```

O ambiente PettingZoo expõe as duas perspectivas para a mesma policy.
**As trajetórias dos dois jogadores entram no rollout PPO**, aproximando de
forma muito mais direta o protocolo de self-play usado por Wang.

Assim, não existe mais:

```text
treinar Gen I contra TypeAware
treinar Gen IV contra TypeAware
treinar Gen IX contra TypeAware
```

como potencial fonte de viés.

Os scripted agents são usados apenas na avaliação.

---

# Common-core benchmark

A v0.6.0 mantém o action space universal:

```text
0–3    move slots
4–9    own team slots
```

Mecânicas específicas como **Terastallization** permanecem fora do benchmark.

Isso é proposital.

A v0.6.0 mede um **núcleo comum de batalha** com exatamente as mesmas ações nas
três gerações.

Uma extensão futura poderá estudar full-mechanics, mas não misturamos isso no
primeiro benchmark entre gerações.

---

# Fixed evaluation pool agora tem 10 pares

Não copie o pool da v0.5.6.

A configuração mudou de:

```text
3 pairs / generation
```

para:

```text
10 pairs / generation
```

Os primeiros 3 pares foram preservados e 7 novos foram adicionados.

Isso melhora muito o desenho estatístico porque a comparação final passa a ter
10 unidades de team-pair por geração.

---

# 1. Instalação

Na pasta da v0.6.0:

```powershell
python -m pip install -r requirements.txt
```

---

# 2. Gere o novo fixed-team pool

Use o mesmo clone do Pokémon Showdown:

```powershell
python prepare_fixed_teams.py `
  --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" `
  --force
```

Depois:

```powershell
Test-Path ".\teams\fixed\manifest.json"
```

deve retornar:

```text
True
```

O `--force` é importante porque a v0.6 usa uma configuração nova com 10 pares.

---

# 3. Valide tudo

```powershell
python scripts\check_crossgen_setup.py
```

O checker confirma:

```text
UBR 1.2
518 observations
10 actions

Gen I / IV / IX
5 seeds
10 fixed pairs/gen

Wang learning-rate schedule
Wang PPO hyperparameters
900k budget
Showdown commit/config hash
```

---

# 4. Inicie o Pokémon Showdown local

Recomendado:

```powershell
cd "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown"

node pokemon-showdown start --no-security
```

Deixe esse terminal aberto.

---

# 5. Smoke test

Em outro PowerShell:

```powershell
conda activate games

cd "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.6.0"

$env:MPLBACKEND="Agg"

python run_crossgen_benchmark.py --smoke
```

O smoke executa:

```text
1 seed
Gen I + IV + IX
8192 steps/gen
checkpoints 4096 / 8192
Wang-style self-play
sparse reward
```

Para não ficar excessivamente lento, o smoke avalia apenas:

```text
pair01
pair02
pair03
```

Os resultados do smoke **não são resultados científicos**.

Ele existe para testar:

```text
Gen I self-play
Gen IV self-play
Gen IX self-play

sparse terminal reward
learning-rate annealing
custom Wang policy architecture
PPO com masking nos logits
checkpoint save/load
controlled mirrored evaluation
```

---

# 6. Experimento completo

Se o smoke passar:

```powershell
$env:MPLBACKEND="Agg"

python run_crossgen_benchmark.py
```

O estudo cria:

```text
crossgen_results\
└── crossgen_YYYYMMDD_HHMMSS\
```

Guarde esse nome.

---

# Checkpoints

Modelos são salvos em:

```text
150k
300k
450k
600k
750k
900k
```

O PPO termina o rollout antes de salvar.

Por isso o projeto guarda:

```text
nominal_checkpoint
actual_model_timesteps
```

separadamente.

---

# Avaliação intermediária

Nos checkpoints não finais:

```text
vs TypeAware
10 pairs
× 2 mirror legs
× 1 repeat

= 20 battles/model/checkpoint
```

Essa avaliação é usada para a curva de aprendizagem entre gerações.

---

# Avaliação final

No 900k:

```text
Random
MaxDamage
TypeAware
SwitchAware
```

Para cada adversário:

```text
10 pairs
× 2 mirror legs
× 3 repeats

= 60 battles / seed / opponent
```

Com 15 modelos:

```text
3600 final battles
```

Mais aproximadamente:

```text
1500 diagnostic battles
```

Total esperado:

```text
5100 controlled evaluation battles
```

---

# Resume

Se o treino parar:

```powershell
python run_crossgen_benchmark.py `
  --study-dir ".\crossgen_results\crossgen_YYYYMMDD_HHMMSS"
```

O programa procura o último checkpoint completo de cada:

```text
generation × seed
```

e continua dali.

Não crie um novo estudo se o objetivo for continuar a mesma execução.

---

# Principais resultados

Ao final:

```text
crossgen_results\
└── crossgen_...\
    ├── study_manifest.json
    ├── runs\
    │   ├── gen1\
    │   ├── gen4\
    │   └── gen9\
    └── aggregate\
        ├── learning_curve.csv
        ├── aggregate_learning_curve.csv
        ├── training_progress.csv
        ├── generation_summary.csv
        ├── generation_comparison.csv
        ├── generation_pairwise_differences.csv
        ├── pair_level_final.csv
        ├── learning_efficiency_typeaware.csv
        └── figures\
```

## `generation_summary.csv`

Resposta principal:

```text
como o mesmo RL terminou em cada geração?
```

separado por adversário.

## `generation_comparison.csv`

Comparações pareadas pelas mesmas seeds:

```text
Gen IV - Gen I
Gen IX - Gen I
Gen IX - Gen IV
```

## `pair_level_final.csv`

Preserva os 10 team pairs.

Esse arquivo é importante para análise estatística posterior porque evita
tratar 60 batalhas repetidas como 60 unidades completamente independentes.

## `learning_efficiency_typeaware.csv`

Resume descritivamente a velocidade de aprendizagem contra TypeAware.

---

# O que NÃO é mais a pergunta principal

A partir daqui não continuamos fazendo:

```text
qual reward é melhor?
qual DQN é melhor?
qual MLP é melhor?
qual curriculum é melhor?
```

a menos que apareça um bug que impeça o benchmark.

A v0.5.x já serviu para validar o pipeline.

A contribuição central passa a ser:

```text
SAME RL
SAME INTERFACE
SAME PROTOCOL

       ↓

GEN I
vs
GEN IV
vs
GEN IX
```

Veja também:

- `docs/LITERATURE_PROTOCOL_v0.6.md`
- `docs/CROSS_GENERATION_DESIGN_v0.6.md`
- `CHANGELOG_v0.6.0.md`
