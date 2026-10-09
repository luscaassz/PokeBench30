# PokéBench30 v0.5.4 — Curriculum Training Study

A v0.5.4 parte dos resultados das versões anteriores:

```text
v0.5      PPO baseline
v0.5.1    UBR 1.2 + battle context
v0.5.2    5-seed UBR 1.1 × UBR 1.2 at 10k
v0.5.3    10k × 50k training-budget study
v0.5.4    curriculum-training study at 300k
```

Na v0.5.3, aumentar apenas o budget de 10k para 50k melhorou o reward interno, mas não produziu melhora consistente no benchmark externo. Trabalhos anteriores de RL em Pokémon usam escalas maiores e/ou adversários de treinamento mais simples, self-play ou curriculum.

A v0.5.4 testa uma única pergunta:

> **Com o mesmo budget total de 300k, a sequência dos adversários de treinamento muda a qualidade da policy final?**

## O que permanece fixo

```text
Generation:       IX
Representation:   UBR 1.2
Observation dim:  518
Action dim:       10
Algorithm:        MaskablePPO
Reward:           igual à v0.5-v0.5.3
Learning rate:    3e-4 constante
PPO architecture: igual à v0.5-v0.5.3
Seeds:            42, 123, 456, 789, 2026
```

Não alteramos reward, UBR ou arquitetura nesta versão para não confundir o efeito do curriculum.

## Regimes padrão

### A — Direct

```text
300k TypeAware
```

### B — Simple curriculum

```text
100k Random
200k TypeAware
```

### C — Progressive curriculum

```text
100k Random
100k MaxDamage
100k TypeAware
```

### D — Random-only control

```text
300k Random
```

## Checkpoints

Cada trajetória salva modelos em:

```text
50k
100k
150k
200k
250k
300k
```

Os checkpoints intermediários são avaliados de forma diagnóstica contra:

```text
Random
MaxDamage
TypeAware
```

Por padrão, cada avaliação intermediária usa apenas 1 repetição por leg:

```text
3 pares × AB/BA × 1 = 6 batalhas/adversário
```

O checkpoint final de 300k usa o protocolo completo:

```text
3 pares × AB/BA × 10 = 60 batalhas/adversário
```

Assim, a curva intermediária serve para visualizar aprendizado; a comparação principal entre regimes deve usar a avaliação final.

## Instalação

```powershell
cd C:\Users\lucas\OneDrive\Desktop\PokeBench_versions\PokeBench30_v0.5.4
python -m pip install -r requirements.txt
```

Se o fixed-team pool ainda não existir nessa pasta:

```powershell
python prepare_fixed_teams.py --showdown-dir "C:\Users\lucas\OneDrive\Desktop\pokemon-showdown" --force
```

Confira o setup:

```powershell
python scripts\check_curriculum_setup.py
```

O esperado inclui:

```text
[OK] PokéBench30 version: 0.5.4
[OK] Representation: ubr-1.2 | obs=518 | actions=10
[OK] Total timesteps/regime: 300000
[OK] Checkpoint interval: 50000
[OK] Curriculum-study setup is ready.
```

## Primeiro: smoke test

Antes das milhões de interações do estudo completo:

```powershell
python run_curriculum_study.py --smoke
```

O smoke executa os quatro regimes com:

```text
1 seed
6,144 timesteps/regime
2,048-step checkpoints
1 repeat/leg
```

Ele serve apenas para validar treinamento, troca de adversário, salvamento, resume, avaliação e agregação. **Não interprete seus scores cientificamente.**

## Estudo completo

```powershell
python run_curriculum_study.py
```

Carga padrão:

```text
4 regimes
× 5 seeds
× 300k nominal timesteps
=
20 trajetórias de PPO
6.0M nominal training steps
```

Como PPO usa rollouts completos de `n_steps=2048`, cada segmento nominal de 50k normalmente executa 51,200 steps. Portanto cada trajetória tende a terminar perto de 307,200 steps reais.

Avaliação aproximada:

```text
por modelo:
5 checkpoints intermediários × 3 adversários × 6 batalhas = 90
checkpoint final × 3 adversários × 60 batalhas             = 180
                                                          -------
                                                            270

20 modelos × 270 = 5,400 batalhas de avaliação
```

Além disso há as batalhas usadas para gerar experiência durante o treinamento.

## Rodar somente os três regimes principais

Se quiser deixar o `random_only` para depois:

```powershell
python run_curriculum_study.py --regimes direct_type_aware random_to_type_aware progressive
```

## Estrutura de resultados

```text
curriculum_results/
└── curriculum_gen9_YYYYMMDD_HHMMSS/
    ├── study_manifest.json
    ├── runs/
    │   ├── direct_type_aware/
    │   │   └── seed_42/
    │   │       ├── checkpoints/
    │   │       │   ├── step_050000/
    │   │       │   ├── ...
    │   │       │   └── step_300000/
    │   │       ├── evaluations/
    │   │       ├── training_logs/
    │   │       └── tensorboard/
    │   └── ...
    └── aggregate/
        ├── learning_curve.csv
        ├── aggregate_learning_curve.csv
        ├── training_progress.csv
        ├── final_summary.csv
        ├── paired_regime_differences.csv
        ├── curriculum_comparison.csv
        └── figures/
```

## Arquivos mais importantes

### `final_summary.csv`

Performance final média de cada regime em 300k.

### `paired_regime_differences.csv`

Para cada seed:

```text
score curriculum - score direct_type_aware
```

### `curriculum_comparison.csv`

Principal tabela de comparação:

```text
regime
opponent
paired_seeds
mean_direct
mean_regime
mean_difference_vs_direct
sample_sd_difference
```

### `aggregate_learning_curve.csv`

Mostra como Random/MaxDamage/TypeAware evoluem ao longo dos checkpoints.

## Resume

Quando o runner iniciar, guarde o caminho mostrado em:

```text
curriculum_results\curriculum_gen9_YYYYMMDD_HHMMSS
```

Se houver interrupção:

```powershell
python run_curriculum_study.py --study-dir "curriculum_results\curriculum_gen9_YYYYMMDD_HHMMSS"
```

O programa:

- reutiliza checkpoints completos;
- recria avaliações ausentes;
- continua o PPO do último checkpoint completo;
- rejeita um `study_dir` criado com configuração diferente.

## Observação sobre resume

A policy e o estado do otimizador do PPO são preservados pelo checkpoint. Porém, a aleatoriedade interna das batalhas do Pokémon Showdown não usa common random numbers entre regimes e uma execução retomada não é garantida como bit-a-bit idêntica a uma execução sem interrupção.

## Próximas ablações possíveis

A v0.5.4 **não** mistura estas mudanças no mesmo experimento:

```text
learning-rate annealing
self-play
recurrent policy
reward v2
state representation v2
MCTS / search
```

Se curriculum não resolver o gap contra TypeAware, esses fatores podem ser avaliados separadamente em versões posteriores.

Veja também:

- `docs/CURRICULUM_v0.5.4.md`
- `docs/LITERATURE_RATIONALE_v0.5.4.md`
- `CHANGELOG_v0.5.4.md`
