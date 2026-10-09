# PokéBench30 v0.4 — Universal Battle Representation

A **v0.4** mantém todo o desenho experimental validado na v0.3 — fixed teams,
mirrored matches, informação parcial e os agentes TypeAware/SwitchAware — e
adiciona uma camada nova entre o Pokémon Showdown e futuros agentes de
Reinforcement Learning:

```text
Pokémon Showdown
      ↓
poke-env Battle
      ↓
Universal Battle Representation (UBR 1.0)
      ↓
vetor fixo + máscara de ações
      ↓
futuro agente de RL
```

A meta desta versão **não é treinar RL ainda**. A meta é garantir que Gen I,
Gen IV e Gen IX possam ser representadas pela mesma interface, sem vazar
Pokémon adversários ocultos.

---

## O que a v0.4 adiciona

### 1. UniversalStateEncoder

Arquivo:

```text
pokebench30/representation/encoder.py
```

Uso:

```python
encoder = UniversalStateEncoder()

observation = encoder.encode(battle)

vector = observation.vector
mask = observation.legal_action_mask
```

Todos os estados produzem:

```text
observation_dim = 472
action_dim      = 10
```

independentemente de a batalha ser Gen I, IV ou IX.

O schema atual é:

```text
UBR 1.0
```

com SHA-256 armazenado em:

```text
config/observation_schema.json
```

Assim, se a ordem ou quantidade de features mudar no futuro, o dataset deixa
uma assinatura explícita da versão usada.

---

# 2. O que existe no vetor de 472 features

A representação é deliberadamente **species-agnostic**.

Ela não contém um one-hot de Pikachu, Charizard, Dragonite etc. Em vez disso,
representa propriedades observáveis compartilhadas entre gerações.

## Estado global

```text
generation_norm
turn_norm
force_switch
own_remaining_fraction
opponent_revealed_fraction
opponent_revealed_remaining_fraction
```

## Boosts dos Pokémon ativos

Para o próprio ativo e para o adversário ativo:

```text
atk
def
spa
spd
spe
accuracy
evasion
```

normalizados para aproximadamente:

```text
[-1, 1]
```

## Próprio time

Existem sempre:

```text
6 slots
```

Cada slot possui:

```text
known
active
fainted
hp_fraction
level_norm
18 tipos one-hot
6 status one-hot
```

## Time adversário

Também existem sempre:

```text
6 slots
```

mas apenas Pokémon **revelados** podem preencher esses slots.

Exemplo:

```text
adversário revelou Dragonite e Gengar

slot 1 → Dragonite
slot 2 → Gengar
slot 3 → UNKNOWN
slot 4 → UNKNOWN
slot 5 → UNKNOWN
slot 6 → UNKNOWN
```

Slots desconhecidos são:

```text
TODOS ZERO
```

Essa é uma regra testada automaticamente.

A UBR nunca tenta reconstruir o time completo a partir da ground truth do
avaliador.

---

# 3. Moves

O Pokémon ativo possui até quatro slots universais:

```text
move_slot_1
move_slot_2
move_slot_3
move_slot_4
```

Cada um contém:

```text
present
legal
base_power_norm
accuracy
priority_norm
type_one_hot_18
category_one_hot_3
```

O encoder não inclui diretamente o score do TypeAware nem o score do
SwitchAware. Isso evita inserir a própria heurística dos baselines dentro da
observação que futuramente será entregue ao agente de RL.

---

# 4. Universal Action Space

A v0.4 define exatamente:

```text
10 ações
```

## Moves

```text
0 → move slot 1
1 → move slot 2
2 → move slot 3
3 → move slot 4
```

## Switches

```text
4 → own team slot 1
5 → own team slot 2
6 → own team slot 3
7 → own team slot 4
8 → own team slot 5
9 → own team slot 6
```

O slot do Pokémon atualmente ativo simplesmente recebe:

```text
mask = 0
```

Exemplo:

```text
legal_action_mask =
[
  1, 1, 0, 0,
  0, 1, 1, 0, 0, 1
]
```

Isso significa que somente aquelas ações podem ser selecionadas.

Durante um forced switch:

```text
mask[0:4] = 0
```

automaticamente.

---

# 5. Por que são 6 ações de switch, e não 5?

Usar seis slots mantém uma associação estável com o próprio time:

```text
team slot 1
team slot 2
...
team slot 6
```

Quando um desses slots contém o Pokémon ativo, ele fica ilegal na máscara.

Isso evita que o significado de uma ação mude apenas porque outro Pokémon
entrou em campo.

---

# 6. O que NÃO entra na UBR 1.0

A primeira representação universal é propositalmente conservadora.

Ainda não codificamos:

```text
species identity
items
abilities
weather
terrain
hazards
volatile effects
Mega Evolution actions
Z-Move actions
Dynamax actions
Terastallization actions
```

Esses elementos não serão adicionados silenciosamente. Uma mudança desse tipo
deverá criar uma nova versão do schema, por exemplo:

```text
UBR 1.1
```

ou:

```text
UBR 2.0
```

Isso é importante para reprodutibilidade.

---

# 7. Estrutura nova

```text
pokebench30/
│
├── representation/
│   ├── __init__.py
│   ├── schema.py
│   ├── action_space.py
│   ├── encoder.py
│   └── validator.py
│
├── agents/
├── evaluation/
├── experiments/
└── teams/
```

Além disso:

```text
config/
├── universal_representation.yaml
└── observation_schema.json
```

---

# 8. Instalação

No Windows + VS Code:

```powershell
python -m pip install -r requirements.txt
```

A v0.4 adiciona:

```text
numpy
```

às dependências.

---

# 9. Reutilizar os fixed teams da v0.3

A v0.4 usa exatamente o mesmo conceito de pool fixo.

Você pode copiar:

```text
PokeBench30_v0.3\teams\fixed
```

para:

```text
PokeBench30_v0.4\teams\fixed
```

ou gerar novamente:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

A configuração da v0.4 está corrigida para:

```text
Gen I  → gen1customgame
Gen IV → gen4customgame
Gen IX → gen9customgame@@@!teampreview
```

Depois:

```powershell
python scripts\check_fixed_pool.py
```

---

# 10. Primeiro teste da representação — sem Showdown

Antes de abrir qualquer batalha:

```powershell
python scripts\check_representation.py
```

Saída esperada:

```text
[OK] UBR schema: ubr-1.0
[OK] Observation dimension: 472
[OK] Action dimension: 10
[OK] Schema SHA-256: ...
[OK] Gen 1: shape=(472,), ...
[OK] Gen 4: shape=(472,), ...
[OK] Gen 9: shape=(472,), ...

[OK] Universal representation offline checks passed.
```

Esse teste verifica que as três gerações têm o mesmo espaço.

---

# 11. Smoke test real da v0.4

Mantenha o servidor Pokémon Showdown aberto.

Depois rode:

```powershell
python run_controlled.py --generations 1 4 9 --matchup type_aware switch_aware --pairs pair01 --repeats 1 --log-universal-state
```

Serão:

```text
3 gerações
×
1 pair
×
2 mirror legs

= 6 batalhas
```

TypeAware e SwitchAware continuam tomando decisões exatamente como antes.

A diferença é que cada action request agora também pode gerar uma observação
UBR.

---

# 12. Novos arquivos de resultado

Com:

```text
--log-universal-state
```

o run também gera:

```text
universal_states.jsonl
observation_schema.json
```

Cada linha do `universal_states.jsonl` contém:

```text
generation
battle_tag
turn
agent

pair_id
mirror_leg
agent_team_id
opponent_team_id

schema_sha256

vector[472]

legal_action_mask[10]

chosen_action_index
chosen_action_label
```

Exemplo conceitual:

```json
{
  "generation": 9,
  "agent": "SwitchAwareAgent",
  "chosen_action_index": 7,
  "chosen_action_label": "switch_team_slot_4",
  "legal_action_mask": [1,1,0,0,0,1,0,1,0,1]
}
```

O vetor é salvo separadamente no mesmo registro.

---

# 13. Validar o dataset real

Depois que o experimento terminar, copie o caminho impresso no terminal:

```text
results\controlled_YYYYMMDD_HHMMSS
```

e rode:

```powershell
python scripts\validate_universal_states.py "results\controlled_YYYYMMDD_HHMMSS"
```

O script verifica:

```text
schema hash
vector dimension
mask dimension
NaN / infinity
selected action mapping
selected action legality
```

Queremos:

```text
[OK] schema mismatches: 0
[OK] dimension errors: 0
[OK] mask errors: 0
[OK] non-finite vectors: 0
[OK] chosen actions marked illegal: 0
```

---

# 14. Information leakage

A política continua sendo a mesma da v0.3:

```text
AGENTE
│
├── próprio time completo
├── oponente ativo
├── oponentes já revelados
└── UNKNOWN nos slots ainda ocultos
```

Enquanto:

```text
AVALIADOR
└── ground truth completa somente após a batalha
```

O encoder não acessa uma estrutura de "full opponent team".

Os testes incluem uma estrutura propositalmente proibida que lança erro se o
encoder tentar consultá-la.

---

# 15. Testes da v0.4

Para executar tudo:

```powershell
python -m unittest discover tests -v
```

A suíte cobre:

```text
fixed teams
mirrored matches
partial information
switch metrics
schema UBR
action mask
forced switch
same shape Gen I / IV / IX
deterministic encoding
zero padding de UNKNOWN
```

---

# 16. Como a v0.4 prepara a v0.5

Na v0.5, o RL não precisará conhecer `poke-env`.

A interface poderá ser aproximadamente:

```python
observation = encoder.encode(battle)

action = policy.predict(
    observation.vector,
    action_mask=observation.legal_action_mask,
)

selected = action_space.object_for_action(
    battle,
    action,
)

return player.create_order(selected)
```

Portanto:

```text
v0.3
controlled experiment
        ↓
v0.4
universal observation/action interface
        ↓
v0.5
reinforcement-learning baseline
        ↓
v0.6
cross-generation transfer
```

---

# 17. Critérios para considerar a v0.4 validada

Antes de começar RL:

```text
[ ] offline representation check passa
[ ] 6-battle live smoke test passa
[ ] Gen I, IV e IX usam shape (472,)
[ ] todas as máscaras têm shape (10,)
[ ] 0 NaN / inf
[ ] 0 hidden-slot leakage
[ ] 0 chosen actions marcadas como ilegais
[ ] schema SHA-256 idêntico nas três gerações
```

Se esses critérios passarem, a representação pode ser congelada como:

```text
UBR 1.0
```

e usada como base da v0.5.
