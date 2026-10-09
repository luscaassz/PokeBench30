# PokéBench30 v0.1

**PokéBench30** é um projeto de pesquisa para estudar agentes de Inteligência Artificial em batalhas Pokémon ao longo de diferentes gerações.

A versão **v0.1** é deliberadamente simples. O objetivo é validar a infraestrutura experimental antes de adicionar Reinforcement Learning, continual learning ou transferência entre gerações.

## Objetivo desta versão

Executar automaticamente batalhas em:

- Generation I — `gen1randombattle`
- Generation IV — `gen4randombattle`
- Generation IX — `gen9randombattle`

usando três agentes básicos:

- **RandomAgent** — escolhe uma ação válida aleatoriamente.
- **MaxDamageAgent** — prioriza o golpe disponível com maior Base Power.
- **TypeAwareAgent** — usa Base Power, precisão, STAB e eficácia de tipo para escolher o golpe ofensivo com maior score.

A saída é salva em CSV para permitir análise posterior.

---

## Estrutura

```text
PokeBench30_v0.1/
├── README.md
├── requirements.txt
├── run_pilot.py
├── config/
│   └── generations.yaml
├── pokebench30/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── random_agent.py
│   │   ├── max_damage_agent.py
│   │   └── type_aware_agent.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── experiments/
│       ├── __init__.py
│       └── pilot.py
├── scripts/
│   └── check_setup.py
├── tests/
│   └── test_metrics.py
└── results/
    └── .gitkeep
```

---

# 1. Requisitos

## Python

Use **Python 3.10 ou superior**.

Verifique:

```bash
python --version
```

No Linux talvez o comando seja:

```bash
python3 --version
```

## Node.js

O Pokémon Showdown atual exige **Node.js 22 ou superior**.

Verifique:

```bash
node --version
```

---

# 2. Criar o ambiente Python

No diretório do projeto:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 3. Instalar o Pokémon Showdown local

Fora deste repositório:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
```

Copie o arquivo de configuração:

### Linux / macOS

```bash
cp config/config-example.js config/config.js
```

### Windows PowerShell

```powershell
Copy-Item config/config-example.js config/config.js
```

Inicie o servidor local:

```bash
node pokemon-showdown start --no-security
```

Por padrão ele ficará disponível em:

```text
http://localhost:8000
```

Mantenha esse terminal aberto.

---

# 4. Verificar a instalação

Abra um **segundo terminal**, entre na pasta do PokéBench30, ative o ambiente virtual e rode:

```bash
python scripts/check_setup.py
```

A saída esperada é parecida com:

```text
[OK] Python compatível
[OK] poke-env importado
[OK] generations.yaml encontrado
[OK] Pokémon Showdown respondeu em localhost:8000
```

---

# 5. Primeiro smoke test

Execute apenas uma batalha na Generation IX:

```bash
python run_pilot.py --generations 9 --battles 1
```

Se funcionar, teste as três gerações:

```bash
python run_pilot.py --generations 1 4 9 --battles 5
```

---

# 6. Rodar o piloto

O piloto padrão executa os três confrontos:

```text
RandomAgent    vs MaxDamageAgent
RandomAgent    vs TypeAwareAgent
MaxDamageAgent vs TypeAwareAgent
```

em cada geração selecionada.

Exemplo:

```bash
python run_pilot.py --generations 1 4 9 --battles 50
```

Para uma primeira execução mais robusta:

```bash
python run_pilot.py --generations 1 4 9 --battles 100
```

Para desativar o log de decisões por turno:

```bash
python run_pilot.py --generations 1 4 9 --battles 100 --no-decision-log
```

---

# 7. Saídas

Cada execução cria uma pasta:

```text
results/run_YYYYMMDD_HHMMSS/
```

com:

### `battles.csv`

Uma linha por batalha.

Exemplo de colunas:

```text
run_id
battle_tag
generation
battle_format
agent_1
agent_2
winner
result_agent_1
turns
agent_1_decisions
agent_2_decisions
```

### `decisions.csv`

Uma linha por decisão tomada por um dos agentes.

Exemplo:

```text
battle_tag
generation
turn
agent
action_type
action
score
```

O `score` só é relevante para agentes heurísticos.

### `summary.csv`

Resumo de cada matchup:

```text
generation
agent_1
agent_2
battles
wins_agent_1
wins_agent_2
draws
win_rate_agent_1
win_rate_agent_2
draw_rate
mean_turns
ci95_low_agent_1
ci95_high_agent_1
```

### `metadata.json`

Metadados da execução, incluindo argumentos utilizados.

---

# 8. Como os agentes funcionam

## RandomAgent

Escolhe uma ação legal aleatoriamente.

Ele é o nosso **baseline mínimo**.

## MaxDamageAgent

Entre os golpes disponíveis, escolhe aquele com maior:

```text
Base Power
```

Se nenhum golpe estiver disponível, escolhe uma ação válida aleatória.

## TypeAwareAgent

Calcula:

```text
score =
    base_power
    × accuracy
    × STAB
    × type_effectiveness
```

onde:

```text
STAB = 1.5 se o tipo do golpe pertence ao Pokémon ativo
       1.0 caso contrário
```

A eficácia é obtida usando a tabela de tipos correspondente à própria geração da batalha.

Esta heurística ainda **não representa a fórmula real de dano**. Ela é propositalmente simples e interpretável para funcionar como baseline.

---

# 9. Por que começar com Random Battle?

Porque o objetivo da v0.1 é validar:

1. comunicação Python ↔ Pokémon Showdown;
2. execução automática em diferentes gerações;
3. arquitetura intercambiável de agentes;
4. logging;
5. métricas;
6. execução em lote.

Não queremos introduzir, ao mesmo tempo, problemas de teambuilding, RL, reward shaping e representação de estado.

---

# 10. Limitações metodológicas da v0.1

Estas limitações são **intencionais**.

## Random teams

Cada batalha de Random Battle pode usar equipes diferentes.

Isso significa que win rate mistura:

- qualidade do agente;
- matchup;
- composição das equipes;
- aleatoriedade da batalha.

Para o piloto isso é aceitável.

Para o artigo final precisaremos controlar melhor essas variáveis.

## Seed

Nesta implementação baseada em `poke-env.Player.battle_against`, não controlamos diretamente a seed interna usada pelo servidor para gerar cada Random Battle.

O projeto registra cada execução, mas ainda não oferece **reprodução exata da sequência de equipes e eventos aleatórios**.

Uma versão futura deverá usar uma das seguintes estratégias:

- geração explícita e armazenada de equipes;
- interface direta com o simulador do Pokémon Showdown;
- configuração de batalhas com seeds conhecidas.

## Mecânicas especiais

Os agentes básicos não tentam usar estrategicamente:

- Mega Evolution;
- Z-Moves;
- Dynamax;
- Terastallization.

Isso será tratado em versões futuras quando expandirmos o benchmark.

## Gerações antigas

O `poke-env` pretende suportar todas as gerações, mas a própria documentação alerta que a biblioteca foi desenvolvida principalmente com gerações recentes em mente.

Portanto, resultados da Generation I devem passar por validação específica antes de qualquer conclusão científica.

---

# 11. Testes locais

Rode:

```bash
python -m unittest discover tests
```

---

# 12. Milestone da v0.1

Consideramos a primeira versão validada quando conseguirmos:

```text
3 agentes
× 3 gerações
× 3 matchups
× pelo menos 100 batalhas por matchup
```

sem erros de infraestrutura.

Isso corresponde a:

```text
900 batalhas
```

no piloto completo.

---

# Próximas versões

## v0.2 — Validação experimental

- repetir cada matchup em múltiplos runs;
- controlar melhor aleatoriedade;
- validar Gen I;
- adicionar bootstrap / intervalos estatísticos;
- gerar gráficos automaticamente;
- salvar versões exatas das dependências.

## v0.3 — Universal State Representation

Criar uma representação que não dependa dos nomes dos Pokémon ou golpes.

## v0.4 — Reinforcement Learning

Adicionar primeiro agente PPO.

## v0.5 — Transfer Experiment

Treinar em uma geração e avaliar em outra.

Objetivo:

```text
             TEST
           G1   G4   G9

TRAIN G1    ●    ●    ●
TRAIN G4    ●    ●    ●
TRAIN G9    ●    ●    ●
```

Essa matriz será o primeiro protótipo do experimento central do PokéBench30.

---

## Nota

Pokémon e Pokémon Showdown são propriedades de seus respectivos detentores. Este repositório é um projeto independente de pesquisa e não é afiliado à The Pokémon Company, Nintendo, Game Freak, Creatures ou Smogon.
