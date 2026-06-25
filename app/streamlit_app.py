"""Dashboard do Brasileirão — Fase 1.

Mostra a tabela do campeonato e a forma recente dos times.
Rodar com:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Permite importar o pacote `src` mesmo rodando via `streamlit run`
RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.data.paths import FORMA_PROC, PARTIDAS_PROC, TABELA_PROC  # noqa: E402
from src.data import preparar_dados  # noqa: E402

st.set_page_config(page_title="Brasileirão Preditor", page_icon="⚽", layout="wide")

CORES_RES = {"V": "#2e7d32", "E": "#9e9e9e", "D": "#c62828"}


# --------------------------------------------------------------------------- #
# Carregamento de dados (com cache e auto-preparação)
# --------------------------------------------------------------------------- #
@st.cache_data
def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # se os dados ainda não foram gerados, roda o pipeline na hora
    if not all(p.exists() for p in (TABELA_PROC, FORMA_PROC, PARTIDAS_PROC)):
        preparar_dados.executar()
    tabela = pd.read_csv(TABELA_PROC)
    forma = pd.read_csv(FORMA_PROC)
    partidas = pd.read_csv(PARTIDAS_PROC)
    return tabela, forma, partidas


def badge_sequencia(seq: str) -> str:
    """Monta os quadradinhos coloridos V/E/D em HTML."""
    bolinhas = []
    for r in str(seq).split("-"):
        cor = CORES_RES.get(r, "#cccccc")
        bolinhas.append(
            f"<span style='display:inline-block;width:26px;height:26px;"
            f"line-height:26px;text-align:center;border-radius:6px;margin:1px;"
            f"color:white;font-weight:700;background:{cor};'>{r}</span>"
        )
    return "".join(bolinhas)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
tabela, forma, partidas = carregar_dados()

st.title("⚽ Brasileirão — Tabela e Forma Recente")
st.caption("Fase 1 · Dashboard de dados. Fonte: API (se configurada) ou simulação.")

# ----- Sidebar: filtro por time -----
times = sorted(tabela["time"].tolist())
with st.sidebar:
    st.header("Filtro")
    time_sel = st.selectbox("Escolha um time", ["(todos)"] + times)
    n_jogos = int(forma["n_jogos"].max())
    st.caption(f"Forma recente calculada sobre os últimos {n_jogos} jogos.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Times", len(tabela))
col2.metric("Partidas", len(partidas))
col3.metric("Rodadas", int(partidas["rodada"].max()))
col4.metric("Gols no campeonato", int(partidas["gols_casa"].sum() + partidas["gols_fora"].sum()))

st.divider()

# ----- Tabela do campeonato -----
st.subheader("📊 Classificação")

tabela_view = tabela.merge(forma[["time", "sequencia"]], on="time", how="left")
if time_sel != "(todos)":
    destaque = tabela_view[tabela_view["time"] == time_sel]
else:
    destaque = tabela_view

# render manual em HTML para colorir a forma
linhas_html = []
for _, r in destaque.iterrows():
    linhas_html.append(
        "<tr>"
        f"<td style='text-align:center'>{int(r.posicao)}º</td>"
        f"<td><b>{r.time}</b></td>"
        f"<td style='text-align:center'>{int(r.P)}</td>"
        f"<td style='text-align:center'>{int(r.J)}</td>"
        f"<td style='text-align:center'>{int(r.V)}</td>"
        f"<td style='text-align:center'>{int(r.E)}</td>"
        f"<td style='text-align:center'>{int(r.D)}</td>"
        f"<td style='text-align:center'>{int(r.GP)}</td>"
        f"<td style='text-align:center'>{int(r.GC)}</td>"
        f"<td style='text-align:center'>{int(r.SG):+d}</td>"
        f"<td>{badge_sequencia(r.sequencia)}</td>"
        "</tr>"
    )

cabecalho = (
    "<tr>"
    + "".join(
        f"<th style='text-align:center;padding:4px 8px'>{c}</th>"
        for c in ["#", "Time", "P", "J", "V", "E", "D", "GP", "GC", "SG", "Forma (5)"]
    )
    + "</tr>"
)
st.markdown(
    f"<table style='width:100%;border-collapse:collapse'>{cabecalho}"
    f"{''.join(linhas_html)}</table>",
    unsafe_allow_html=True,
)

st.divider()

# ----- Forma recente em destaque -----
st.subheader("🔥 Forma recente")

if time_sel != "(todos)":
    f = forma[forma["time"] == time_sel].iloc[0]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric(
            f"Pontos nos últimos {int(f.n_jogos)} jogos",
            f"{int(f.pontos_ultimos_n)} pts",
            f"{int(f.vitorias)}V {int(f.empates)}E {int(f.derrotas)}D",
        )
        st.metric("Gols (pró / contra)", f"{int(f.gols_pro)} / {int(f.gols_contra)}")
    with c2:
        st.markdown("**Sequência (antigo → recente):**")
        st.markdown(badge_sequencia(f.sequencia), unsafe_allow_html=True)
else:
    # grade de cards com a sequência de cada time
    cols = st.columns(4)
    for i, (_, f) in enumerate(forma.iterrows()):
        with cols[i % 4]:
            st.markdown(f"**{f.time}** — {int(f.pontos_ultimos_n)} pts")
            st.markdown(badge_sequencia(f.sequencia), unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

st.divider()

# ----- Gráfico de pontos -----
st.subheader("📈 Pontuação por time")
graf = tabela.sort_values("P", ascending=True)
fig = px.bar(
    graf,
    x="P",
    y="time",
    orientation="h",
    text="P",
    labels={"P": "Pontos", "time": ""},
    height=600,
)
fig.update_traces(marker_color="#1565c0", textposition="outside")
st.plotly_chart(fig, width="stretch")
