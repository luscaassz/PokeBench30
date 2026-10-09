# PokéBench30 v0.2.1 — Partial Observation + Experimental Instrumentation

**PokéBench30** é um benchmark de pesquisa para estudar agentes de IA em batalhas Pokémon ao longo de diferentes gerações.

Esta revisão da **v0.2.1** corrige a política de informação do benchmark: um agente conhece integralmente o próprio time, mas conhece do adversário **apenas os Pokémon que já foram revelados pela batalha**, assim como ocorre para um jogador humano.

## Princípio central: sem information leakage

Durante `choose_move()`, o agente pode usar:

```text
próprio time completo
+ Pokémon adversário ativo
+ Pokémon adversários já revelados
+ histórico/estado observado da batalha
```

Ele **não recebe** os slots ainda ocultos do oponente.

O benchmark mantém duas camadas diferentes:

### 1. Observação do agente

Usada durante a tomada de decisão e registrada em `actions.csv`.

Exemplos:

```text
own_team_size
own_remaining
opponent_revealed
opponent_revealed_remaining
opponent_revealed_hp_sum
```

### 2. Ground truth do avaliador

Calculada **somente depois da batalha** a partir do time próprio de cada jogador.

É salva em `battles.csv` para avaliação, mas nunca é passada aos agentes.

Exemplos:

```text
agent_1_team_size_true
agent_2_team_size_true
agent_1_remaining_true
agent_2_remaining_true
agent_1_hp_sum_true
agent_2_hp_sum_true
```

Também registramos quanto cada jogador chegou a revelar do adversário:

```text
agent_1_opponent_revealed
agent_2_opponent_revealed
agent_1_opponent_reveal_fraction
agent_2_opponent_reveal_fraction
```

Isso permite estudar posteriormente **opponent modeling sob informação parcial**.

---

## Agentes

### RandomAgent

Escolhe uniformemente entre golpes normais e switches legais.

### MaxDamageAgent

Prioriza o golpe disponível com maior Base Power.

### TypeAwareAgent

Usa:

```text
score = base_power × accuracy × STAB × type_effectiveness
```

### SwitchAwareAgent

Usa a heurística ofensiva do TypeAwareAgent, mas também compara o matchup atual com os Pokémon disponíveis para troca.

Pode trocar quando:

- a troca é obrigatória;
- o matchup atual é desfavorável e existe melhora suficiente;
- o HP está muito baixo e a troca melhora o matchup.

Nenhum dos quatro baselines ativa deliberadamente Mega Evolution, Z-Moves, Dynamax ou Terastallization nesta versão. As flags dessas mecânicas continuam registradas para auditoria.

---

# Arquivos gerados

```text
results/
└── run_YYYYMMDD_HHMMSS/
    ├── battles.csv
    ├── actions.csv
    ├── summary.csv
    ├── agent_summary.csv
    ├── metadata.json
    └── figures/
        ├── agent_win_rate.png
        ├── switch_rate.png
        └── voluntary_switch_rate.png
```

## `actions.csv`

Uma linha por **action request**, não necessariamente uma por turno.

O estado é capturado antes da ação.

Campos importantes:

```text
battle_tag
generation
turn
request_index
agent

active_pokemon
active_hp_fraction
active_type_1
active_type_2

opponent_pokemon
opponent_hp_fraction
opponent_type_1
opponent_type_2

own_team_size
own_remaining

opponent_revealed
opponent_revealed_remaining

action_type
action
voluntary_switch
forced_switch
decision_reason
heuristic_score
```

Os campos `opponent_revealed*` são sempre a visão parcial daquele agente.

## `battles.csv`

Uma linha por batalha.

Há duas famílias de campos:

### Ground truth — apenas avaliação

```text
agent_1_team_size_true
agent_2_team_size_true
agent_1_remaining_true
agent_2_remaining_true
agent_1_fainted_true
agent_2_fainted_true
agent_1_hp_sum_true
agent_2_hp_sum_true
```

### Informação revelada a cada agente

```text
agent_1_opponent_revealed
agent_2_opponent_revealed
agent_1_opponent_revealed_remaining
agent_2_opponent_revealed_remaining
agent_1_opponent_reveal_fraction
agent_2_opponent_reveal_fraction
```

Por exemplo:

```text
agent_1_team_size_true = 6
agent_2_team_size_true = 6

agent_1_opponent_revealed = 2
```

significa:

> O avaliador sabe que o agente 2 possuía seis Pokémon, mas o agente 1 só chegou a ver dois deles.

Isso é comportamento esperado, não erro.

## `agent_summary.csv`

Além de win rate e switches, inclui:

```text
mean_opponent_reveal_fraction
```

para medir quanto do time adversário ficou conhecido durante as partidas.

---

# Instalação — Windows + VS Code + Conda

Dentro da pasta do projeto:

```powershell
conda run -n games python -m pip install -r requirements.txt
```

Mantenha o Pokémon Showdown aberto em outro terminal:

```powershell
node pokemon-showdown start --no-security
```

Verifique:

```powershell
conda run -n games python scripts\check_setup.py
```

---

# Smoke test

Primeiro:

```powershell
conda run -n games python run_pilot.py --generations 9 --battles 1
```

Depois:

```powershell
conda run -n games python run_pilot.py --generations 1 4 9 --battles 5
```

Com quatro agentes, o segundo comando executa:

```text
3 gerações × 6 matchups × 5 = 90 batalhas
```

---

# O que verificar no smoke test

Em `battles.csv`, é normal observar algo como:

```text
agent_1_team_size_true = 6
agent_2_team_size_true = 6
agent_1_opponent_revealed = 2
agent_2_opponent_revealed = 4
```

Isso confirma a separação entre:

```text
GROUND TRUTH DO AVALIADOR
```

e

```text
INFORMAÇÃO PARCIAL DO AGENTE
```

Em `actions.csv`, nenhum campo deve revelar Pokémon adversários ainda desconhecidos.

---

# Limitações atuais

- Random Battle ainda não possui seed controlada neste fluxo.
- Os times aleatórios ainda confundem habilidade do agente com matchup de equipes.
- Fixed teams + mirrored matches continuam planejados para a v0.3.
- SwitchAware é uma heurística transparente, não um modelo completo de busca.
- Gerações antigas ainda precisam de validação específica.
- Mecânicas especiais de geração serão incorporadas explicitamente em uma versão posterior.

---

# Próximo objetivo

Depois de validar esta v0.2.1:

```text
v0.3
├── fixed teams
├── mirrored matches
├── controlled experimental pairs
└── stronger reproducibility
```

A política de informação parcial desta versão deve permanecer nas versões futuras.
