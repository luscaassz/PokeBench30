# PokéBench30 v0.3 — Fixed Teams + Mirrored Matches

A v0.3 transforma o PokéBench30 de um piloto baseado em Random Battle em um
**experimento controlado por times fixos e partidas espelhadas**.

A v0.2.2 continua disponível no projeto por meio de `run_pilot.py`, mas o novo
experimento principal é executado com:

```powershell
python run_controlled.py
```

---

## O problema que a v0.3 resolve

No Random Battle:

```text
Agente A recebe Team X
Agente B recebe Team Y
```

e em outra batalha:

```text
Agente A recebe Team Z
Agente B recebe Team W
```

Isso mistura duas coisas:

```text
qualidade do agente
+
qualidade/matchup dos times
```

Na v0.3, cada confronto usa um par fixo:

```text
TEAM A
vs
TEAM B
```

e executa duas pernas:

```text
LEG AB
Agent 1 + Team A
vs
Agent 2 + Team B

LEG BA
Agent 1 + Team B
vs
Agent 2 + Team A
```

Isso é um **mirrored match**.

---

# 1. Times fixos reproduzíveis

Os times não são escritos manualmente.

Eles são gerados pelo próprio Pokémon Showdown usando:

```text
random-battle generator
+
seed explícita
```

Exemplo conceitual:

```text
gen1randombattle
seed = 101,202,303,404
        ↓
packed team
        ↓
congelado em teams/fixed/
```

O Pokémon Showdown fornece geração de times por seed, e a v0.3 salva:

```text
seed
packed team
export humano
SHA-256
Showdown git commit
Node version
```

Assim, depois de preparado, o experimento nunca depende de um novo Random Team.

---

# 2. Por que gerar com Random Battle e lutar em Custom Game?

O gerador de Random Battle fornece automaticamente sets coerentes com a geração.

Depois de gerados, os times são congelados e enviados explicitamente aos jogadores.

As batalhas controladas usam:

```text
gen1customgame@@@!teampreview
gen4customgame@@@!teampreview
gen9customgame@@@!teampreview
```

A remoção de Team Preview preserva a política de informação parcial:

```text
próprio time completo
+
oponentes já revelados
+
slots adversários ocultos continuam ocultos
```

---

# 3. Estrutura

```text
PokeBench30_v0.3/
│
├── prepare_fixed_teams.py
├── run_controlled.py
├── run_pilot.py
│
├── config/
│   ├── generations.yaml
│   └── fixed_teams.yaml
│
├── pokebench30/
│   ├── agents/
│   ├── evaluation/
│   │   ├── analysis.py
│   │   ├── information.py
│   │   ├── metrics.py
│   │   └── mirrored.py
│   │
│   ├── experiments/
│   │   ├── pilot.py
│   │   └── controlled.py
│   │
│   └── teams/
│       └── fixed_pool.py
│
├── teams/
│   └── fixed/        # criado pelo prepare_fixed_teams.py
│
└── results/
```

---

# 4. Preparar os times pela primeira vez

Mantenha seu repositório local do Pokémon Showdown instalado.

No VS Code, dentro de `PokeBench30_v0.3`:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\CAMINHO\pokemon-showdown"
```

Exemplo, se o Showdown estiver no Desktop:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown"
```

O script irá gerar **18 times**:

```text
3 gerações
×
3 pares
×
2 times
=
18 times fixos
```

Depois:

```powershell
python scripts\check_fixed_pool.py
```

Saída esperada:

```text
[OK] Fixed-team manifest found
[OK] 18 team records
[OK] Showdown commit: ...
[OK] All packed-team SHA-256 hashes match
```

---

# 5. O que é salvo em `teams/fixed/`

Exemplo:

```text
teams/fixed/
├── manifest.json
├── gen1/
│   ├── pair01_A.packed.txt
│   ├── pair01_A.txt
│   ├── pair01_B.packed.txt
│   ├── pair01_B.txt
│   └── ...
├── gen4/
└── gen9/
```

`.packed.txt` é o formato enviado ao simulador.

`.txt` é a versão legível por humanos.

Não altere manualmente esses arquivos após criar o pool. O projeto verifica os hashes.

---

# 6. Smoke test da v0.3

Com o servidor Pokémon Showdown aberto:

```powershell
python run_controlled.py --generations 9 --matchup type_aware switch_aware --pairs pair01 --repeats 1
```

Isso executa somente:

```text
1 geração
×
1 par de times
×
2 pernas espelhadas
×
1 repetição
=
2 batalhas
```

---

# 7. Primeiro piloto controlado

Depois:

```powershell
python run_controlled.py --generations 1 4 9 --matchup type_aware switch_aware --repeats 5
```

Com os três pares fixos:

```text
3 gerações
×
3 pares
×
2 pernas
×
5 repetições
=
90 batalhas
```

---

# 8. Experimento maior

Quando o piloto estiver validado:

```powershell
python run_controlled.py --generations 1 4 9 --matchup type_aware switch_aware --repeats 50
```

Total:

```text
3 × 3 × 2 × 50 = 900 batalhas
```

Agora essas 900 batalhas usam **os mesmos times fixos**, alternados entre os agentes.

---

# 9. Null test

A v0.3 permite self-play do mesmo agente:

```powershell
python run_controlled.py --generations 1 4 9 --matchup type_aware type_aware --repeats 10
```

Isso funciona como teste do próprio harness.

Com agentes idênticos e times espelhados, esperamos que a pontuação controlada tenda a:

```text
0.5
```

Não precisa ser exatamente 0.5 em uma amostra pequena devido à aleatoriedade da batalha.

---

# 10. Resultados novos

Cada execução controlada cria:

```text
results/
└── controlled_YYYYMMDD_HHMMSS/
    ├── battles.csv
    ├── actions.csv
    ├── summary.csv
    ├── agent_summary.csv
    ├── mirror_cycles.csv
    ├── pair_summary.csv
    ├── controlled_summary.csv
    ├── metadata.json
    │
    ├── teams_snapshot/
    │   ├── manifest.json
    │   ├── gen1_pair01_A.packed.txt
    │   └── ...
    │
    └── figures/
        ├── agent_win_rate.png
        ├── switch_rate.png
        ├── voluntary_switch_rate.png
        ├── mirrored_score.png
        └── team_assignment_gap.png
```

---

# 11. `mirror_cycles.csv`

Cada linha representa:

```text
mesmo par de times
+
mesmos dois agentes
+
uma batalha em cada orientação
```

Exemplo:

```text
AB:
TypeAware usa Team A
SwitchAware usa Team B

BA:
TypeAware usa Team B
SwitchAware usa Team A
```

O Agent 1 recebe:

```text
1 ponto por vitória
0.5 por empate
0 por derrota
```

Então um ciclo pode resultar em:

```text
2.0  → venceu as duas
1.5  → vitória + empate
1.0  → ciclo equilibrado
0.5
0.0
```

---

# 12. Métrica principal da v0.3

A nova métrica é:

```text
normalized_mirror_score_agent_1
```

calculada como:

```text
pontos obtidos
------------------------------
2 × quantidade de mirror cycles
```

Interpretação:

```text
0.50 → equilíbrio
>0.50 → Agent 1 obteve mais pontos
<0.50 → Agent 2 obteve mais pontos
```

Ela deve ser interpretada junto com tamanho de amostra e variabilidade, não como
uma pontuação absoluta de inteligência.

---

# 13. Diagnóstico de viés de time

Também registramos:

```text
agent_1_win_rate_as_team_a
agent_1_win_rate_as_team_b
agent_1_team_assignment_gap
```

onde:

```text
assignment gap =
|WR usando Team A - WR usando Team B|
```

Se esse valor for alto, o par de times ainda está interagindo fortemente com o agente.

Além disso:

```text
team_a_wins
team_b_wins
team_a_win_rate_all
```

permitem identificar pares de times muito desequilibrados.

---

# 14. Reprodutibilidade

Cada run guarda uma cópia exata dos times usados em:

```text
teams_snapshot/
```

e registra:

```text
team seed
team SHA-256
fixed-team config SHA-256
Pokémon Showdown git commit
Node version
poke-env version
Python version
```

Portanto, mesmo se `teams/fixed/` for modificado no futuro, cada resultado continua
associado aos bytes exatos dos times daquele experimento.

---

# 15. Informação parcial

A política da v0.2.2 continua válida.

Os agentes não recebem a ground truth do time adversário.

Durante a batalha:

```text
battle.team
→ próprio time

battle.opponent_team
→ somente informação revelada pelo simulador
```

A ground truth completa é usada somente após o fim da batalha para métricas.

---

# 16. Limitação que ainda permanece

A v0.3 controla:

```text
✓ composição dos times
✓ seeds de geração dos times
✓ atribuição A/B por espelhamento
✓ versão do Showdown
✓ hashes dos times
```

Mas ainda não controla diretamente:

```text
✗ RNG interno de cada batalha
```

Por exemplo:

```text
accuracy
critical hits
damage rolls
secondary effects
speed ties
```

O próximo passo de reprodutibilidade total exigiria controlar o simulador em um
nível mais baixo, diretamente pelo BattleStream / simulator protocol.

---

# Milestone da v0.3

Consideramos a v0.3 validada quando:

```text
1. os 18 times são preparados e os hashes passam;
2. smoke test de 2 batalhas funciona;
3. null test fica aproximadamente centrado em 0.5;
4. TypeAware × SwitchAware roda nas três gerações;
5. cada mirror cycle contém exatamente AB + BA;
6. nenhum slot oculto é vazado aos agentes.
```

Depois disso, o PokéBench30 estará pronto para começar a transição para a
representação universal e, depois, Reinforcement Learning.
