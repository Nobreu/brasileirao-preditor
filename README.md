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
- **Fase 1 — Dashboard** ✅ (você está aqui)
- Fase 2 — Feature engineering e EDA
- Fase 3 — Modelo preditivo
- Fase 4 — API com FastAPI
- Fase 5 — Camada de contexto + deploy
