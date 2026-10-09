# PokéBench30 v0.6.1 — Zero-Shot Transfer + Behavioral Telemetry

> **Hotfix 2 (fixed-pool aggregate):** corrige o último problema encontrado no smoke do Hotfix 1.
> Os catálogos estruturais do fixed pool agora usam **exclusivamente** as linhas imutáveis
> geradas dos arquivos `packed_fixed_team`; observações pós-batalha não podem mais
> contaminar `fixed_pool_pokemon_catalog.csv` nem `fixed_pool_move_catalog.csv`.
>
> O Hotfix 2 também mantém as três correções do Hotfix 1:
> 1. catálogo amplo de Pokémon limitado a espécies-base canônicas por geração;
> 2. resolução canônica de registros duplicados do manifest;
> 3. boosts `SELF` classificados como `self_boost`.
>
> **Nenhum modelo da v0.6.0 precisa ser retreinado.** Refazer somente o smoke/avaliação
> v0.6.1 regenera os catálogos e resumos corretos.

Antes do smoke, valide o hotfix:

```powershell
python scripts\check_telemetry_hotfix.py
```

O checker agora exige, além das validações anteriores:

```text
Gen 1 fixed pool -> 120 Pokémon
Gen 4 fixed pool -> 120 Pokémon
Gen 9 fixed pool -> 120 Pokémon
nenhuma chave duplicada generation+pair+side+team_slot
nenhuma chave duplicada generation+pair+side+team_slot+move_slot
nenhum team_slot 7-12 no catálogo estrutural agregado
nenhuma linha battle_observation vazando para fixed_pool_*
```

A v0.6.1 mantém **exatamente os modelos treinados na v0.6.0**.

Não há novo treinamento, novo reward, nova arquitetura ou novo algoritmo.

A nova pergunta é:

> **O conhecimento aprendido em uma geração transfere para outra e, quando o comportamento muda, quais aspectos estratégicos mudam exatamente?**

## Experimento principal

```text
TREINO ↓ / TESTE →   Gen I   Gen IV   Gen IX
Gen I                   ✓       ✓        ✓
Gen IV                  ✓       ✓        ✓
Gen IX                  ✓       ✓        ✓
```

Os 15 modelos finais de 900k da v0.6.0 são reutilizados sem fine-tuning.

A diagonal também é reavaliada porque queremos que as nove células tenham a **mesma telemetria**.

## Adversários

Por padrão:

```text
MaxDamage
TypeAware
```

Eles são os dois adversários principais porque a v0.6.0 mostrou espaço útil entre piso e teto.

Para incluir também:

```text
Random
SwitchAware
```

use `--all-opponents`.

## Carga

Estudo principal:

```text
9 train/eval cells
× 5 seeds
× 2 opponents
× 10 pairs
× 2 mirror legs
× 3 repeats
=
5.400 battles
```

Com os quatro oponentes:

```text
10.800 battles
```

## Métrica de transferência

Além do score absoluto, a v0.6.1 calcula:

```text
transfer gap(A → B)
=
score(model trained A, tested B)
-
score(model trained B, tested B)
```

sempre pareando pela mesma seed.

## Telemetria robusta

A v0.6.1 separa quatro camadas:

```text
O QUE EXISTE NA GERAÇÃO (dex do poke-env)
        ↓
O QUE EXISTE NO POOL FIXO DO BENCHMARK
        ↓
O QUE ERA LEGAL NAQUELE TURNO
        ↓
O QUE O AGENTE ESCOLHEU
```

Isso evita interpretar frequência bruta como preferência.

Para golpes, armazenamos:

```text
move id
move type
physical / special / status
base power
accuracy
priority
STAB
type effectiveness
```

mais tags multi-label:

```text
damaging
status
self buff
self debuff
target buff
target debuff
any buff
any debuff
status infliction
volatile status
healing
drain
recoil
hazard set / remove
weather
terrain
priority
protect-like
pivot-like
force switch
```

Um golpe pode possuir várias tags simultaneamente.

Também são registrados todos os golpes **legais disponíveis** no momento da decisão.

Assim conseguimos calcular:

```text
availability_rate
selected_rate
selection_given_available
```

para:

```text
um golpe específico
um tipo de golpe
physical/special/status
buff/debuff
hazards
healing
priority
etc.
```

## Tipagem dos Pokémon

A telemetria inclui:

```text
active Pokémon type 1 / type 2
opponent active type 1 / type 2
switch target type 1 / type 2
```

Além disso, o benchmark cria um catálogo deduplicado dos times fixos de cada geração.

Isso permite comparar, por exemplo:

```text
Gen I tem mais Psychic no pool?
Gen IX oferece mais tipos diferentes?
O RL escolhe golpes Water porque prefere Water
ou simplesmente porque Water aparece mais no pool?
```

## Catálogo estrutural da geração

A v0.6.1 também lê diretamente `GenData.from_gen(...)` do poke-env e produz:

```text
generation_pokemon_dex_catalog.csv
generation_move_dex_catalog.csv
generation_dex_overview.csv
generation_pokemon_dex_type_summary.csv
generation_move_dex_type_summary.csv
generation_move_dex_category_summary.csv
generation_move_dex_feature_summary.csv
```

Isso permite testar perguntas como:

```text
A proporção de golpes damaging cresce com a geração?
Há mais golpes de buff/debuff disponíveis nas gerações recentes?
A diversidade de tipos dos Pokémon aumentou?
Quais tipos de golpe existem com maior frequência em cada geração?
```

Essa camada descreve os dados por geração fornecidos pelo poke-env, independentemente do agente.

## Catálogo do pool fixo

A v0.6.1 gera:

```text
fixed_pool_pokemon_catalog.csv
fixed_pool_move_catalog.csv
fixed_pool_pokemon_type_summary.csv
fixed_pool_move_type_summary.csv
fixed_pool_move_category_summary.csv
fixed_pool_move_feature_summary.csv
fixed_pool_move_presence_summary.csv
```

Esses arquivos descrevem os **20 times congelados usados na avaliação de cada geração**.

Eles não são apresentados como uma descrição completa do metagame inteiro da geração.

---

# 1. Copie o fixed pool da v0.6.0

A configuração dos 10 pares não mudou.

Na pasta v0.6.1:

```powershell
Remove-Item ".\teams\fixed" -Recurse -Force -ErrorAction SilentlyContinue

Copy-Item `
  "C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.6.0\teams\fixed" `
  ".\teams\fixed" `
  -Recurse
```

## 2. Descubra o diretório do estudo completo da v0.6.0

Ele é o diretório que contém:

```text
study_manifest.json
runs\gen1
runs\gen4
runs\gen9
aggregate\generation_summary.csv
```

Exemplo:

```text
C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.6.0\crossgen_results\crossgen_YYYYMMDD_HHMMSS
```

## 3. Checker

```powershell
python scripts\check_transfer_setup.py `
  --source-study-dir "C:\...\crossgen_YYYYMMDD_HHMMSS"
```

Ele valida:

```text
UBR 1.2
fixed pool
Showdown provenance
3 generations
5 seeds
15 final 900k models
metadata de cada model
```

## 4. Showdown

Em outro terminal:

```powershell
cd "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown"
node pokemon-showdown start --no-security
```

## 5. Smoke

```powershell
$env:MPLBACKEND="Agg"

python run_transfer_telemetry.py `
  --source-study-dir "C:\...\crossgen_YYYYMMDD_HHMMSS" `
  --smoke
```

O smoke usa:

```text
seed 42
3 × 3 transfer matrix
MaxDamage apenas
pair01–pair03
1 repeat/leg
```

Ele existe apenas para validar cross-generation inference + telemetry.

## 6. Estudo principal

```powershell
$env:MPLBACKEND="Agg"

python run_transfer_telemetry.py `
  --source-study-dir "C:\...\crossgen_YYYYMMDD_HHMMSS"
```

Para os quatro oponentes:

```powershell
python run_transfer_telemetry.py `
  --source-study-dir "C:\...\crossgen_YYYYMMDD_HHMMSS" `
  --all-opponents
```

## Resume

```powershell
python run_transfer_telemetry.py `
  --source-study-dir "C:\...\crossgen_YYYYMMDD_HHMMSS" `
  --study-dir ".\transfer_results\transfer_YYYYMMDD_HHMMSS"
```

Avaliações completas já existentes são reutilizadas.

---

# Principais resultados

```text
transfer_results\transfer_...\aggregate\
```

## Transferência

```text
transfer_scores_by_seed.csv
transfer_matrix_summary.csv
transfer_gap_by_seed.csv
transfer_gap_summary.csv
zero_shot_generalization_summary.csv
figures\transfer_heatmap_*.png
figures\transfer_gap_heatmap_*.png
```

## Telemetria

```text
aggregate\telemetry\
    behavior_by_seed.csv
    behavior_summary.csv
    move_by_seed.csv
    move_summary.csv
    move_type_by_seed.csv
    move_type_summary.csv
    move_category_summary.csv
    move_feature_summary.csv
    pokemon_type_summary.csv
    generation_dex_overview.csv
    generation_move_dex_catalog.csv
    generation_pokemon_dex_catalog.csv
    generation_move_dex_feature_summary.csv
    fixed_pool_pokemon_catalog.csv
    fixed_pool_move_catalog.csv
    fixed_pool_pokemon_type_summary.csv
    fixed_pool_move_type_summary.csv
    fixed_pool_move_category_summary.csv
    fixed_pool_move_feature_summary.csv
    fixed_pool_move_presence_summary.csv
```

Cada avaliação individual ainda preserva:

```text
actions.csv
telemetry_move_opportunities.csv.gz
```

para análises posteriores sem precisar repetir batalhas.

Mais detalhes:

- `docs/TRANSFER_TELEMETRY_v0.6.1.md`
- `docs/LITERATURE_PROTOCOL_v0.6.md`
- `docs/CROSS_GENERATION_DESIGN_v0.6.md`
