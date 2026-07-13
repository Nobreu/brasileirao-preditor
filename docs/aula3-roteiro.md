# Aula 3 — Modelo Preditivo

## Objetivo
Treinar um modelo de classificação que preveja o resultado de uma partida
(Casa / Empate / Fora), validado com **split temporal** para não vazar dados.

## Conceitos
- Classificação multiclasse
- Split treino/teste **temporal** (não aleatório)
- Regressão Logística e Random Forest
- Métricas: acurácia, matriz de confusão, log loss
- Baseline ingênuo
- Importância de features
- Persistência de modelo com joblib

## Passo a passo (2h)

| Tempo | Etapa | Arquivo |
|-------|-------|---------|
| 15 min | Setup + revisão do dataset da Fase 2 | — |
| 20 min | Split temporal (por que não embaralhar) | `src/models/treinar.py` |
| 30 min | Baseline + Regressão Logística | `src/models/treinar.py` |
| 25 min | Random Forest + importância de features | `src/models/treinar.py` |
| 20 min | Persistência + função de previsão + aba no app | `src/models/predizer.py`, `app/streamlit_app.py` |
| 10 min | Commit + reflexão ("dá pra apostar?") | — |

## Como rodar

```bash
git checkout fase-3-modelo
python -m src.data.preparar_dados     # garante o dataset
python -m src.models.treinar          # treina e salva models/modelo_v1.pkl
streamlit run app/streamlit_app.py    # abra a aba "🧠 Previsão (ML)"
```

## Metodologia (resumo)
1. **Split temporal:** as ~30% rodadas mais recentes viram teste; o resto é treino.
2. **Baseline ingênuo:** "o mandante sempre vence" (~48%). Régua a superar.
3. **Dois modelos:** Regressão Logística (linear, interpretável) e Random Forest
   (robusto, não-linear). O melhor por acurácia é salvo.
4. **Avaliação:** acurácia, log loss e matriz de confusão no conjunto de teste.
5. **Previsão ao vivo:** `prever_partida(casa, fora)` monta as features do estado
   atual dos times (mesmo formato do treino) e devolve as probabilidades.

## Entregável
- `models/modelo_v1.pkl` salvo.
- Aba "🧠 Previsão (ML)" no dashboard, comparando o modelo com o simulador de Poisson.
- Métricas documentadas.

## Critério de aceitação
- [x] Split temporal implementado corretamente (testado)
- [x] Matriz de confusão gerada
- [x] Aba de previsão funcionando
- [~] Modelo supera o acaso (33%); bater o baseline do mandante depende dos dados

## Reflexão (para a turma)
> "O modelo é bom o suficiente para apostar? Por que não?" — poucos dados,
> empates são difíceis de prever, e o futebol é imprevisível **de propósito**.
> O valor está na **metodologia correta**, não em acertar muito.

## Gancho para a Aula 4
> "O modelo só funciona se rodarmos o script. Próxima aula: vamos transformar
> isso num **serviço de verdade**, com uma API."
