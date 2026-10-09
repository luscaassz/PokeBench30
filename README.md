# PokéBench30 v0.5 — MaskablePPO RL Baseline

A v0.5 adiciona o primeiro agente de Reinforcement Learning ao PokéBench30,
mantendo congelada a representação validada na v0.4.1:

```text
UBR 1.1
observation = 472
action space = 10
```

O objetivo desta versão não é ainda estudar transferência entre gerações.
Primeiro queremos provar que uma política aprendida consegue:

```text
UBR 1.1
   ↓
MaskablePPO
   ↓
ação entre 10 possibilidades
   ↓
legal-action mask
   ↓
Pokémon Showdown
```

e depois participar do mesmo benchmark controlado de fixed teams + mirrored
matches criado na v0.3.

---

## O que há de novo

```text
pokebench30/
└── rl/
    ├── env.py
    ├── mask_adapter.py
    ├── model_io.py
    ├── policy_agent.py
    ├── reward.py
    ├── training.py
    └── analysis.py
```

Novos comandos:

```text
train_rl.py
run_rl_controlled.py
analyze_rl_training.py
scripts/check_rl_setup.py
```

Documentação técnica:

```text
docs/RL_BASELINE_v0.5.md
```

---

# 1. Algoritmo

A baseline é:

```text
MaskablePPO
```

via `sb3-contrib`.

Por que usar action masking?

Em um turno, várias ações da UBR não são legais. Exemplos:

```text
move desabilitado
Pokémon desmaiado
slot do Pokémon atualmente ativo
forced switch
```

A máscara garante que a política escolha apenas entre ações válidas.

---

# 2. Representação congelada

A v0.5 não altera a UBR.

```text
UBR 1.1
472 features
10 actions
```

O modelo salva no `metadata.json`:

```text
schema version
schema SHA-256
observation dimension
action dimension
```

Se o código atual não corresponder ao schema usado no treinamento, o modelo é
recusado.

---

# 3. Treino e avaliação usam distribuições diferentes

## Treinamento

A política aprende em:

```text
gen1randombattle
gen4randombattle
ou
gen9randombattle
```

Isso fornece diversidade de times.

## Avaliação

A avaliação usa:

```text
fixed teams
+
mirrored assignments
```

da v0.3.

Assim:

```text
Random Battle = distribuição de treino

Fixed/Mirrored = benchmark controlado
```

---

# 4. Reward da baseline

O reward inicial é deliberadamente simples:

```text
vitória               30.0
Pokémon desmaiado      2.0
HP                      1.0
status                  0.5
```

Ele utiliza o `reward_computing_helper` do `poke-env`, que retorna a diferença
de valor entre estados consecutivos.

O objetivo da v0.5 é validar o pipeline, não fazer reward engineering.

---

# 5. Instalação

No VS Code / PowerShell:

```powershell
cd C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5
```

Instale:

```powershell
python -m pip install -r requirements.txt
```

As dependências novas principais são:

```text
gymnasium
stable-baselines3
sb3-contrib
```

Depois:

```powershell
python scripts\check_rl_setup.py
```

Queremos:

```text
[OK] poke-env imported
[OK] gymnasium imported
[OK] stable-baselines3 imported
[OK] sb3-contrib imported
[OK] torch imported
[OK] UBR schema: ubr-1.1
[OK] Observation dimension: 472
[OK] Action dimension: 10

[OK] PokéBench30 v0.5 RL setup is ready.
```

---

# 6. Pokémon Showdown

Deixe seu servidor local rodando como nas versões anteriores.

O treinamento usa o servidor porque cada episódio é uma batalha real simulada
pelo Pokémon Showdown.

Os fixed teams não são necessários para TREINAR.

Eles só serão necessários para a avaliação controlada posterior.

---

# 7. Primeiro smoke training

Comece pequeno.

```powershell
python train_rl.py --generation 9 --opponent type_aware --timesteps 1024 --n-steps 256 --batch-size 64
```

Esse comando serve apenas para verificar:

```text
ambiente abre
batalhas acontecem
MaskablePPO coleta experiências
gradientes são calculados
modelo é salvo
```

Não interprete o desempenho científico desse treinamento curto.

A saída será algo como:

```text
models/
└── maskable_ppo_gen9_YYYYMMDD_HHMMSS/
    ├── maskable_ppo.zip
    ├── metadata.json
    ├── training_config.json
    ├── monitor.csv.monitor.csv
    ├── checkpoints/
    └── tensorboard/
```

---

# 8. Primeiro treinamento real

Depois que o smoke training funcionar:

```powershell
python train_rl.py --generation 9 --opponent type_aware --timesteps 10000
```

Depois podemos aumentar progressivamente:

```text
10k
50k
100k
```

mas não recomendo começar com 100k antes de validar o aprendizado.

---

# 9. O que é um timestep aqui?

Um timestep de PPO não significa uma batalha.

Ele corresponde a uma decisão do agente:

```text
estado
→ ação
→ novo estado
```

Logo:

```text
10.000 timesteps
```

representam milhares de decisões distribuídas entre várias batalhas.

---

# 10. Analisar o treinamento

Depois:

```powershell
python analyze_rl_training.py "models\maskable_ppo_gen9_YYYYMMDD_HHMMSS"
```

Isso cria:

```text
figures/
├── training_reward.png
└── training_episode_length.png
```

Não use apenas a curva de reward para afirmar que o agente ficou melhor.
A avaliação contra agentes externos continua sendo necessária.

---

# 11. Preparar o benchmark fixo

Se `teams/fixed/manifest.json` não existir na v0.5:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

Depois:

```powershell
python scripts\check_fixed_pool.py
```

---

# 12. Primeiro teste controlado do RL

Suponha que seu modelo esteja em:

```text
models\maskable_ppo_gen9_20260928_120000\maskable_ppo.zip
```

Rode:

```powershell
python run_rl_controlled.py ^
  --model "models\maskable_ppo_gen9_20260928_120000\maskable_ppo.zip" ^
  --generation 9 ^
  --opponent type_aware ^
  --pairs pair01 ^
  --repeats 1
```

No PowerShell você também pode colocar tudo em uma linha:

```powershell
python run_rl_controlled.py --model "models\maskable_ppo_gen9_20260928_120000\maskable_ppo.zip" --generation 9 --opponent type_aware --pairs pair01 --repeats 1
```

Isso gera exatamente:

```text
LEG AB
RL + Team A
vs
TypeAware + Team B

LEG BA
RL + Team B
vs
TypeAware + Team A
```

---

# 13. Avaliação maior

Depois:

```powershell
python run_rl_controlled.py --model "CAMINHO_DO_MODELO" --generation 9 --opponent type_aware --repeats 10
```

Sem `--pairs`, usa:

```text
pair01
pair02
pair03
```

Total:

```text
3 pares
×
2 mirror legs
×
10 repetições
=
60 batalhas
```

A principal métrica continua sendo:

```text
normalized_mirror_score_agent_1
```

Na avaliação v0.5:

```text
Agent 1 = RLPolicyAgent
```

portanto:

```text
> 0.5 → RL obteve mais pontos
≈ 0.5 → equilíbrio
< 0.5 → baseline obteve mais pontos
```

sempre considerando tamanho de amostra e variação entre pares.

---

# 14. Opponents disponíveis

Treino e avaliação aceitam:

```text
random
max_damage
type_aware
switch_aware
```

O default de treino é:

```text
type_aware
```

Uma progressão razoável para experimentos futuros seria:

```text
Random
  ↓
TypeAware
  ↓
SwitchAware
```

Mas a v0.5 usa um oponente fixo por treinamento.

---

# 15. Gen I e Gen IV

A arquitetura funciona para:

```powershell
python train_rl.py --generation 1 --opponent type_aware --timesteps 10000
```

e:

```powershell
python train_rl.py --generation 4 --opponent type_aware --timesteps 10000
```

Cada modelo, porém, deve ser avaliado na mesma geração em que foi treinado.

Exemplo aceito:

```text
train Gen IV
test Gen IV
```

Exemplo rejeitado na v0.5:

```text
train Gen I
test Gen IX
```

Esse experimento é propositalmente reservado para:

```text
PokéBench30 v0.6
Cross-Generation Transfer
```

---

# 16. Por que bloquear transferência agora?

Porque queremos separar:

```text
v0.5
o RL funciona?
```

de:

```text
v0.6
o conhecimento aprendido transfere entre gerações?
```

Se misturarmos as duas perguntas agora, fica difícil saber se uma falha veio do
pipeline de RL ou da transferência.

---

# 17. Arquitetura

```text
              TRAINING
                 │
                 ▼
        Random Battle Gen IX
                 │
                 ▼
              UBR 1.1
        observation[472]
        legal mask[10]
                 │
                 ▼
            MaskablePPO
                 │
                 ▼
         maskable_ppo.zip


              EVALUATION
                 │
                 ▼
         maskable_ppo.zip
                 │
                 ▼
           RLPolicyAgent
                 │
                 ▼
       Fixed + Mirrored Teams
                 │
                 ▼
    RL vs TypeAware / SwitchAware
```

---

# 18. Critério para considerar a v0.5 validada

A v0.5 não exige que o RL já seja campeão.

Ela está validada quando:

```text
[ ] check_rl_setup passa
[ ] smoke training termina
[ ] model.zip é salvo
[ ] metadata.json é salvo
[ ] schema UBR 1.1 é validado ao carregar
[ ] policy nunca seleciona ação mascarada
[ ] modelo pode ser recarregado
[ ] RLPolicyAgent completa batalhas
[ ] avaliação mirrored funciona
[ ] training curves são geradas
```

Depois disso, fazemos múltiplas seeds e um experimento de baseline mais sério.

Só então passamos para:

```text
v0.6 — Cross-Generation Transfer
```

---

## Organização científica até aqui

```text
v0.1   infraestrutura
v0.2   scripted baselines + logging
v0.2.2 strategic switching estável
v0.3   fixed teams + mirrored matches
v0.4.1 UBR 1.1 validada
v0.5   MaskablePPO baseline
v0.6   cross-generation transfer
```
