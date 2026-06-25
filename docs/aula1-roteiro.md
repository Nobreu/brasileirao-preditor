# Aula 1 — Dashboard de Tabela e Forma Recente

## Objetivo
Ao final da aula o aluno tem um app Streamlit funcionando que mostra a tabela
do Brasileirão e a forma recente dos times.

## Conceitos
- Coleta de dados (API ou simulação/raspagem) com `requests`
- Manipulação de DataFrames com Pandas
- Visualização com Plotly
- App com Streamlit
- Git básico

## Passo a passo (2h)

| Tempo | Etapa | Arquivo |
|-------|-------|---------|
| 15 min | Setup (venv + `pip install -r requirements.txt`) | — |
| 30 min | Coleta de dados | `src/data/coletar_tabela.py` |
| 30 min | Tratamento (nomes, datas, nulos) | `src/data/limpar.py` |
| 30 min | Forma recente (últimas 5) | `src/data/forma.py` + `notebooks/01-eda.ipynb` |
| 15 min | Dashboard | `app/streamlit_app.py` |

## Como rodar

```bash
python -m venv venv
# Windows:  venv\Scripts\activate     |  Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 1) gerar os dados (coleta -> limpeza -> forma)
python -m src.data.preparar_dados

# 2) subir o dashboard
streamlit run app/streamlit_app.py
```

> Sem chave de API, os dados são gerados por uma **simulação determinística**
> (mesma semente → mesmos números), então a aula roda offline. Para usar dados
> reais, defina `FOOTBALL_DATA_API_KEY` antes de rodar a coleta.

## Critério de aceitação
- [x] Dados de pelo menos uma rodada em `data/processed/`
- [x] App roda sem erro com `streamlit run app/streamlit_app.py`
- [x] Tabela mostra todos os 20 times
- [x] Forma recente aparece como sequência visual (cores V/E/D)
- [ ] Código commitado e pushado (feito pelos alunos)

## Gancho para a Aula 2
> "Vocês mostraram **o que aconteceu**. Para prever **o que vai acontecer**,
> precisamos transformar isso em variáveis que um modelo entenda.
> Próxima aula: feature engineering."
