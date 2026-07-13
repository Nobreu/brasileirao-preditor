# Brasileirão Preditor ⚽

Projeto de Data Science em 5 fases: do dashboard à previsão de resultados do
Campeonato Brasileiro. **Você está na Fase 1.**

## Fase 1 — Dashboard de Tabela e Forma Recente
App Streamlit que coleta os dados do campeonato, calcula a forma recente de
cada time e exibe tudo num painel interativo.

## Como rodar

```bash
# 1) ambiente
python -m venv venv
# Windows:  venv\Scripts\activate   |  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 2) gerar os dados (coleta -> limpeza -> forma recente)
python -m src.data.preparar_dados

# 3) subir o dashboard
streamlit run app/streamlit_app.py
```

O dashboard também gera os dados sozinho na primeira abertura, caso você pule o
passo 2.

### Dados reais (opcional)
Sem chave de API o projeto usa uma **simulação determinística** (a aula roda
offline). Para dados reais de [football-data.org](https://www.football-data.org):

```bash
# Windows (PowerShell):  $env:FOOTBALL_DATA_API_KEY = "sua-chave"
# Linux/Mac:             export FOOTBALL_DATA_API_KEY="sua-chave"
python -m src.data.preparar_dados
```

## Estrutura

```
brasileirao-preditor/
├── app/streamlit_app.py        # dashboard (Fase 1+)
├── src/data/
│   ├── coletar_tabela.py       # coleta (API ou simulação)
│   ├── limpar.py               # tratamento dos dados
│   ├── forma.py                # cálculo da forma recente
│   ├── preparar_dados.py       # pipeline completo
│   ├── times.py                # times + padronização de nomes
│   └── paths.py                # caminhos centrais
├── notebooks/01-eda.ipynb      # exploração
├── tests/test_dados.py         # testes simples
├── data/{raw,processed}/       # dados (gerados, fora do Git)
└── docs/aula1-roteiro.md       # roteiro da aula
```

## Testes
```bash
python tests/test_dados.py      # ou: python -m pytest -q
```

## Roadmap
- **Fase 1 — Dashboard** ✅
- **Fase 2 — Feature engineering e EDA** ✅
- **Fase 3 — Modelo preditivo** ✅ (você está aqui)
- Fase 4 — API com FastAPI
- Fase 5 — Camada de contexto + deploy

### Fase 3 — o que foi adicionado
- `src/models/treinar.py` — split temporal, baseline, Regressão Logística + Random Forest, métricas (acurácia, log loss, matriz de confusão), salva `models/modelo_v1.pkl`
- `src/models/predizer.py` — `prever_partida()` usando o estado atual dos times
- Aba **"🧠 Previsão (ML)"**: probabilidades do modelo, comparação ML × Poisson, desempenho, matriz de confusão e importância das features
- `tests/test_modelo.py`, `notebooks/03-modelo.ipynb`, [`docs/aula3-explicacao-professor.md`](docs/aula3-explicacao-professor.md)
- Treinar: `python -m src.models.treinar`

### Fase 2 — o que foi adicionado
- `src/features/build_features.py` — gera `dataset_modelagem.csv` (1 linha/partida, 18 features), **sem data leak**
- Nova aba **"Análise de Times"** no dashboard (features, comparação, scatter casa×fora, heatmap de correlação)
- `notebooks/02-features.ipynb` — EDA (heatmap, boxplot, scatter)
- `tests/test_features.py` — inclui teste automático de data leak

### Fase 2+ — Simulador (modelo de Poisson)
- `src/features/simulador.py` — simulador de partidas e projeção de temporada
- Aba **"🎲 Simulador"**: probabilidades de resultado, placar mais provável, gols esperados, BTTS/Over 2.5, heatmap de placares, narração automática e Monte Carlo (10k)
- Aba **"🏆 Projeção"**: Monte Carlo do resto do campeonato → chance de título / G4 / rebaixamento (usa os jogos futuros da API)
- `tests/test_simulador.py` — somas de probabilidade e invariantes da projeção
- Detalhes em [`docs/aula2-simulador.md`](docs/aula2-simulador.md)
