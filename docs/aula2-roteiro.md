# Aula 2 — Feature Engineering e EDA

## Objetivo
Transformar os dados brutos em **features** que descrevem o estado de cada time
**antes** de cada partida, produzindo o `dataset_modelagem.csv` que vai alimentar
o modelo na Aula 3.

## Conceitos
- Feature engineering
- Análise exploratória (EDA) avançada
- Janelas móveis (rolling windows) — "últimos N jogos"
- **Data leak** e como evitá-lo (o conceito central da aula)
- Visualização comparativa (heatmap, boxplot, scatter)

## Passo a passo (2h)

| Tempo | Etapa | Arquivo |
|-------|-------|---------|
| 15 min | Trocar de branch + revisão da Aula 1 | — |
| 45 min | Construir features (uma a uma, explicando o porquê) | `src/features/build_features.py` |
| 30 min | EDA (heatmap, boxplot, scatter) | `notebooks/02-features.ipynb` |
| 20 min | Atualizar o dashboard (aba "Análise de Times") | `app/streamlit_app.py` |
| 10 min | Commit + reflexão | — |

## Como rodar

```bash
git checkout fase-2-features
python -m src.data.preparar_dados     # agora também gera o dataset de features
streamlit run app/streamlit_app.py    # abra a aba "Análise de Times"
```

## Features geradas (18 no total)
Para o mandante (`casa_`) e o visitante (`fora_`):
- **forma5** — pontos nos últimos 5 jogos
- **aprov_mando** — aproveitamento no mando específico (casa p/ mandante, fora p/ visitante)
- **media_gols_pro5 / media_gols_sofr5** — média de gols marcados/sofridos (últimos 5)
- **saldo_temporada** — saldo de gols acumulado
- **sem_derrota / sem_vitoria** — sequência atual invicta / sem ganhar

Comparativas e de contexto:
- **diff_forma5, diff_saldo, diff_aprov_mando** — diferença mandante − visitante
- **h2h_saldo_casa** — confronto direto (saldo de vitórias do mandante no histórico)

Alvo: **`alvo`** ∈ {Casa, Empate, Fora}.

## O ponto mais importante: SEM DATA LEAK
As features de uma partida só usam jogos **anteriores** a ela. Garantimos isso
construindo o dataset em ordem cronológica: calculamos as features com o
histórico acumulado e **só depois** registramos o resultado do jogo atual.
Há um teste automático que prova isso (`tests/test_features.py::test_sem_data_leak`).

## Critério de aceitação
- [x] `dataset_modelagem.csv` tem pelo menos 8 features além do alvo (tem 18)
- [x] Não há data leak (verificado por teste automático em 3 partidas)
- [x] EDA produziu pelo menos 3 visualizações (heatmap, boxplot, scatter)
- [x] Dashboard tem nova aba funcionando

## Gancho para a Aula 3
> "Agora a gente tem o **input certo**. Próxima aula, a gente ensina a máquina a
> aprender com isso — e descobre quais dessas features realmente importam."
