# PokéBench30 v0.2 — Experimental Instrumentation

**PokéBench30** é um benchmark de pesquisa para estudar agentes de IA em batalhas Pokémon ao longo de diferentes gerações.

A v0.1 provou que o mesmo pipeline consegue executar Gen I, Gen IV e Gen IX. A **v0.2** transforma esse protótipo em uma infraestrutura experimental mais informativa.

## O que mudou na v0.2

- Quarto agente: **SwitchAwareAgent**.
- `decisions.csv` virou **`actions.csv`**: uma chamada de `choose_move()` é uma *action request*, não necessariamente um turno.
- Estado da batalha é capturado antes de cada ação.
- Logging de Pokémon ativos, HP, tipos, status, boosts e ações legais.
- Separação de **switch voluntário** e **switch forçado**.
- Métricas de Pokémon restantes e soma de HP no fim da batalha.
- Intervalo de confiança de Wilson e teste binomial exploratório por matchup.
- `agent_summary.csv` criado automaticamente.
- Gráficos automáticos em `figures/`.
- Versão do `poke-env` registrada no metadata.

---

## Agentes

### RandomAgent
Escolhe qualquer ação legal aleatoriamente. É o baseline mínimo.

### MaxDamageAgent
Prioriza o golpe disponível com maior Base Power.

### TypeAwareAgent
Usa a heurística:

```text
score = base_power × accuracy × STAB × type_effectiveness
```

### SwitchAwareAgent
Mantém a heurística ofensiva do TypeAwareAgent, mas compara o matchup atual com os Pokémon disponíveis para troca.

Ele pode trocar quando:

- a troca é obrigatória;
- o matchup atual é desfavorável e existe uma melhora suficientemente grande;
- o HP está muito baixo e uma troca melhora o matchup.

O objetivo **não** é criar um agente ótimo, mas um baseline interpretável que possua decisão explícita `attack vs switch`.

---

# Instalação no Windows + VS Code + Conda

Você já pode continuar usando o ambiente `games` sem precisar de `conda activate`:

```powershell
conda run -n games python -m pip install -r requirements.txt
```

Mantenha o Pokémon Showdown local aberto em outro terminal:

```powershell
cd C:\Users\lucas\OneDrive\Desktop\pokemon-showdown
node pokemon-showdown start --no-security
```

Verifique o ambiente:

```powershell
conda run -n games python scripts\check_setup.py
```

---

# Primeiro smoke test da v0.2

Teste primeiro somente Gen IX e 1 batalha por matchup:

```powershell
conda run -n games python run_pilot.py --generations 9 --battles 1
```

Como existem 4 agentes, há 6 combinações:

```text
Random × MaxDamage
Random × TypeAware
Random × SwitchAware
MaxDamage × TypeAware
MaxDamage × SwitchAware
TypeAware × SwitchAware
```

Logo, esse comando executa 6 batalhas.

Depois teste as três gerações:

```powershell
conda run -n games python run_pilot.py --generations 1 4 9 --battles 5
```

Isso produz:

```text
3 gerações × 6 matchups × 5 = 90 batalhas
```

---

# Piloto recomendado

Depois do smoke test:

```powershell
conda run -n games python run_pilot.py --generations 1 4 9 --battles 50
```

Total:

```text
3 × 6 × 50 = 900 batalhas
```

Não é necessário rodar esse lote antes de validar o smoke test.

---

# Arquivos gerados

Cada execução cria:

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
        └── switch_rate.png
```

## battles.csv

Uma linha por batalha, com:

- resultado;
- duração;
- action requests;
- golpes;
- switches;
- switches voluntários;
- switches forçados;
- Pokémon conhecidos/restantes;
- soma das frações de HP restantes.

## actions.csv

Uma linha por solicitação de ação. Inclui o estado **antes** da ação:

```text
battle_tag
generation
turn
request_index
agent
active_pokemon
active_hp_fraction
active_status
active_type_1
active_type_2
active_boosts
opponent_pokemon
opponent_hp_fraction
opponent_status
opponent_type_1
opponent_type_2
opponent_boosts
action_type
action
voluntary_switch
forced_switch
decision_reason
heuristic_score
move_base_power
move_accuracy
move_type
move_stab
move_type_effectiveness
...
```

## summary.csv

Resumo por matchup e geração, incluindo:

- win rate;
- draws;
- duração média;
- Wilson 95% CI;
- teste binomial bilateral exploratório contra 50%;
- Pokémon restantes médios.

**Importante:** o p-value desta versão é diagnóstico/exploratório. Ele não corrige múltiplas comparações e não elimina os vieses dos Random Battles.

## agent_summary.csv

Resume cada agente por geração:

- batalhas;
- vitórias/derrotas/empates;
- win rate;
- duração média;
- action requests;
- movimentos;
- switches;
- switch rate;
- switches voluntários/forçados.

---

# Analisar novamente um run existente

```powershell
conda run -n games python run_analysis.py results\run_YYYYMMDD_HHMMSS
```

Para não gerar figuras:

```powershell
conda run -n games python run_analysis.py results\run_YYYYMMDD_HHMMSS --no-plots
```

---

# Rodar apenas alguns agentes

Exemplo, comparar TypeAware e SwitchAware:

```powershell
conda run -n games python run_pilot.py --generations 1 4 9 --agents type_aware switch_aware --battles 50
```

Isso é útil para experimentos específicos sem rodar todos os pares.

---

# Limitações ainda presentes

A v0.2 melhora **instrumentação**, não o controle experimental.

Ainda usamos Random Battle. Portanto:

1. os times mudam entre partidas;
2. o matchup dos times influencia o resultado;
3. a seed interna do Random Battle não é controlada neste workflow;
4. os dois agentes não recebem necessariamente os mesmos times em condições espelhadas;
5. o SwitchAware é uma heurística simples de tipos, não um planejador ótimo;
6. Gen I precisa continuar sendo validada cuidadosamente.

Esses pontos são justamente o foco da **v0.3: Experimental Control**.

---

# Roadmap

```text
v0.1  Infrastructure validation        ✓
v0.2  Experimental instrumentation     ← atual
v0.3  Fixed teams + mirrored matches
v0.4  Universal state representation
v0.5  RL baseline (PPO)
v0.6  Cross-generation transfer matrix
```

## Próximo objetivo científico

A v0.2 deve permitir responder não apenas **quem ganha**, mas **como os agentes jogam**:

- quais agentes trocam mais;
- quais fazem switches voluntários;
- que tipos de matchup desencadeiam troca;
- quantos Pokémon restam no fim;
- quanto tempo cada estilo leva para vencer;
- como essas características mudam entre Gen I, IV e IX.

---

Pokémon e Pokémon Showdown pertencem aos seus respectivos detentores. Este projeto é independente, acadêmico e não afiliado à The Pokémon Company, Nintendo, Game Freak, Creatures ou Smogon.
