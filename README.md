# PokéBench30 v0.5.1 — UBR 1.2 Battle Context

A v0.5.1 é uma atualização **controlada de representação** sobre a v0.5.

O algoritmo de RL continua o mesmo:

```text
MaskablePPO
```

A principal mudança é:

```text
UBR 1.1: 472 observações
        ↓
UBR 1.2: 518 observações
```

O espaço de ações continua:

```text
10 ações
```

---

## Por que esta versão existe?

A v0.5 validou o pipeline completo:

```text
Pokémon Showdown
→ poke-env
→ UBR
→ action mask
→ MaskablePPO
→ modelo salvo
→ RLPolicyAgent
→ fixed + mirrored evaluation
```

O modelo Gen IX de 10k timesteps obteve:

```text
vs RandomAgent:    45-15   mirror score = 0.750
vs TypeAwareAgent:  5-55   mirror score = 0.083
```

Ao analisar os logs apareceram repetições de ações como:

```text
Sticky Web
Sticky Web
Sticky Web

Stealth Rock
Stealth Rock
Stealth Rock

Taunt
Taunt
Taunt
```

A UBR 1.1 não informava à MLP se esses efeitos persistentes já estavam ativos.

A v0.5.1 testa exatamente essa hipótese.

---

# UBR 1.2

```text
schema_version  = ubr-1.2
observation_dim = 518
action_dim      = 10

SHA-256:
d4b3858aab3a1c3bb1e17d9f34becf93814be05bfc949d52023651035f20e4c3
```

Os primeiros **472 valores** preservam exatamente o layout da UBR 1.1.

Depois são adicionadas **46 features de contexto**:

```text
16 side conditions
 8 weather
 6 field / terrain
16 active volatile effects
--------------------------
46 novas features
```

Logo:

```text
472 + 46 = 518
```

---

# Contexto incluído

## Para os dois lados

```text
Stealth Rock
Spikes
Toxic Spikes
Sticky Web
Reflect
Light Screen
Aurora Veil
Tailwind
```

Spikes:

```text
0 layers → 0.0
1 layer  → 0.333
2 layers → 0.667
3 layers → 1.0
```

Toxic Spikes:

```text
0 layers → 0.0
1 layer  → 0.5
2 layers → 1.0
```

Os demais são presença binária.

## Weather

```text
Rain
Sun
Sandstorm
Hail
Snow
Primordial Sea
Desolate Land
Delta Stream
```

## Field

```text
Electric Terrain
Grassy Terrain
Misty Terrain
Psychic Terrain
Trick Room
Gravity
```

## Efeitos nos Pokémon ativos

Para o próprio Pokémon ativo e para o adversário ativo:

```text
Taunt
Encore
Substitute
Confusion
Leech Seed
Disable
Yawn
Protect
```

---

# O que continua de fora

UBR 1.2 ainda é uma baseline compacta.

Não codificamos:

```text
species identity
items
abilities
PP
memória de moves revelados do adversário
todos os efeitos raros do Showdown
Mega / Z-Move / Dynamax / Tera no action space
```

Qualquer expansão futura deve gerar uma nova versão da UBR.

---

# Instalação

Entre na pasta:

```powershell
cd C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.1
```

Instale:

```powershell
python -m pip install -r requirements.txt
```

A v0.5.1 agora inclui também:

```text
tensorboard>=2.17
```

porque o smoke test real da v0.5 mostrou que essa dependência era necessária.

---

# 1. Verificar ambiente

```powershell
python scripts\check_rl_setup.py
```

Esperado:

```text
[OK] poke-env imported
[OK] gymnasium imported
[OK] stable-baselines3 imported
[OK] sb3-contrib imported
[OK] torch imported
[OK] tensorboard imported
[OK] numpy imported
[OK] PyYAML imported
[OK] PokéBench30 version: 0.5.1
[OK] UBR schema: ubr-1.2
[OK] Observation dimension: 518
[OK] Action dimension: 10
```

---

# 2. Teste offline da representação

```powershell
python scripts\check_representation.py
```

Esperado:

```text
[OK] UBR schema: ubr-1.2
[OK] Observation dimension: 518
[OK] Action dimension: 10
...
[OK] Universal representation offline checks passed.
```

No sample offline da Gen IX o script também ativa artificialmente:

```text
Stealth Rock
Spikes
Sticky Web
Rain
Electric Terrain
Taunt
```

para verificar o novo bloco de contexto.

---

# 3. Preparar os fixed teams

Como nas versões anteriores:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

Depois:

```powershell
python scripts\check_fixed_pool.py
```

---

# 4. Smoke test REAL da UBR 1.2

Antes de treinar, rode os agentes heurísticos com logging universal:

```powershell
python run_controlled.py --generations 1 4 9 --matchup type_aware switch_aware --pairs pair01 --repeats 1 --log-universal-state
```

Isso gera 6 batalhas:

```text
Gen I  → AB + BA
Gen IV → AB + BA
Gen IX → AB + BA
```

Depois:

```powershell
python scripts\validate_universal_states.py "results\controlled_XXXXXXXX_XXXXXX"
```

Queremos:

```text
Chosen actions mapped into UBR: X/X

schema mismatches: 0
dimension errors: 0
mask errors: 0
non-finite vectors: 0
battle-context value errors: 0
empty legal-action masks: 0
unmapped chosen actions: 0
chosen actions marked illegal: 0
```

Esse é o primeiro gate da v0.5.1.

---

# 5. Modelos da v0.5 NÃO são compatíveis

Isso é proposital.

Um modelo salvo com:

```text
UBR 1.1
472 features
```

não pode ser carregado como:

```text
UBR 1.2
518 features
```

O metadata checker deve recusar o modelo antigo.

Não copie `maskable_ppo_gen9_20260928_115136` para usar como modelo v0.5.1.

Ele permanece como **baseline UBR 1.1**.

---

# 6. Primeiro smoke training

Depois que a UBR 1.2 passar no teste real:

```powershell
python train_rl.py --generation 9 --opponent type_aware --timesteps 1024 --n-steps 256 --batch-size 64 --seed 42
```

O objetivo é apenas validar:

```text
518 features
→ PPO
→ mask
→ treino
→ save
→ reload
```

---

# 7. Experimento comparável à v0.5

Quando o smoke funcionar, use a mesma configuração principal da baseline:

```powershell
python train_rl.py --generation 9 --opponent type_aware --timesteps 10000 --seed 42
```

Mantemos:

```text
MaskablePPO
learning_rate = 0.0003
n_steps = 2048
batch_size = 128
network = [256, 128]
reward igual
opponent igual
seed igual
budget igual
```

Mudança principal:

```text
UBR 1.1 → UBR 1.2
```

---

# 8. Analisar treinamento

```powershell
python analyze_rl_training.py "models\maskable_ppo_gen9_XXXXXXXX_XXXXXX"
```

---

# 9. Avaliação contra RandomAgent

```powershell
python run_rl_controlled.py --model "models\maskable_ppo_gen9_XXXXXXXX_XXXXXX\maskable_ppo.zip" --generation 9 --opponent random --repeats 10
```

60 batalhas.

Baseline UBR 1.1:

```text
45-15
mirror score = 0.750
```

---

# 10. Avaliação contra TypeAwareAgent

```powershell
python run_rl_controlled.py --model "models\maskable_ppo_gen9_XXXXXXXX_XXXXXX\maskable_ppo.zip" --generation 9 --opponent type_aware --repeats 10
```

60 batalhas.

Baseline UBR 1.1:

```text
5-55
mirror score = 0.083
```

---

# Comparação principal

Depois teremos:

```text
                 Random        TypeAware
UBR 1.1           0.750          0.083
UBR 1.2              ?              ?
```

Esse primeiro experimento ainda usa uma única seed.

Se houver sinal de melhoria, o próximo passo correto é repetir várias seeds,
não declarar conclusão a partir de apenas seed 42.

---

# Arquitetura

```text
                    UBR 1.2
        ┌─────────────────────────────┐
        │ UBR 1.1 core       472     │
        │                             │
        │ side conditions     16     │
        │ weather              8     │
        │ field                6     │
        │ volatile effects    16     │
        └─────────────────────────────┘
                    │
                    ▼
              observation[518]
                    │
              legal mask[10]
                    │
                    ▼
                MaskablePPO
```

---

# Organização científica

```text
v0.1     infraestrutura
v0.2     scripted agents + logging
v0.2.2   switching estável
v0.3     fixed teams + mirrored evaluation
v0.4.1   UBR 1.1 estruturalmente validada
v0.5     primeiro MaskablePPO + baseline UBR 1.1
v0.5.1   UBR 1.2 battle-context ablation
v0.6     cross-generation transfer
```

Mais detalhes:

```text
docs/UBR_1_2.md
docs/RL_BASELINE_v0.5.1.md
docs/V0_5_BASELINE.md
CHANGELOG_v0.5.1.md
```
