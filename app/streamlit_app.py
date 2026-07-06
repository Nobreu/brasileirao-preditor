"""Dashboard do Brasileirão.

Fase 1: tabela do campeonato e forma recente dos times.
Fase 2: aba de análise de times (features, comparação, correlações).

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

from src.data.paths import (  # noqa: E402
    DATASET_MODELAGEM,
    FORMA_PROC,
    PARTIDAS_PROC,
    TABELA_PROC,
)
from src.data import preparar_dados  # noqa: E402
from src.features.build_features import resumo_times  # noqa: E402

st.set_page_config(page_title="Brasileirão Preditor", page_icon="⚽", layout="wide")

CORES_RES = {"V": "#2e7d32", "E": "#9e9e9e", "D": "#c62828"}


# --------------------------------------------------------------------------- #
# Carregamento de dados (com cache e auto-preparação)
# --------------------------------------------------------------------------- #
@st.cache_data
def carregar_dados():
    if not all(
        p.exists() for p in (TABELA_PROC, FORMA_PROC, PARTIDAS_PROC, DATASET_MODELAGEM)
    ):
        preparar_dados.executar()
    tabela = pd.read_csv(TABELA_PROC)
    forma = pd.read_csv(FORMA_PROC)
    partidas = pd.read_csv(PARTIDAS_PROC)
    dataset = pd.read_csv(DATASET_MODELAGEM)
    resumo = resumo_times(partidas)
    return tabela, forma, partidas, dataset, resumo


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
# Dados + cabeçalho
# --------------------------------------------------------------------------- #
tabela, forma, partidas, dataset, resumo = carregar_dados()

st.title("⚽ Brasileirão — Painel de Dados")
st.caption("Fases 1 e 2 · Fonte: API football-data.org (se configurada) ou simulação.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Times", len(tabela))
col2.metric("Partidas", len(partidas))
col3.metric("Rodadas", int(partidas["rodada"].max()))
col4.metric(
    "Gols no campeonato",
    int(partidas["gols_casa"].sum() + partidas["gols_fora"].sum()),
)

aba_tabela, aba_analise = st.tabs(["📊 Classificação & Forma", "🔬 Análise de Times"])

# =========================================================================== #
# ABA 1 — Classificação e forma recente (Fase 1)
# =========================================================================== #
with aba_tabela:
    times = sorted(tabela["time"].tolist())
    time_sel = st.selectbox("Filtrar por time", ["(todos)"] + times, key="filtro_tab")

    st.subheader("📊 Classificação")
    tabela_view = tabela.merge(forma[["time", "sequencia"]], on="time", how="left")
    if time_sel != "(todos)":
        destaque = tabela_view[tabela_view["time"] == time_sel]
    else:
        destaque = tabela_view

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
        cols = st.columns(4)
        for i, (_, f) in enumerate(forma.iterrows()):
            with cols[i % 4]:
                st.markdown(f"**{f.time}** — {int(f.pontos_ultimos_n)} pts")
                st.markdown(badge_sequencia(f.sequencia), unsafe_allow_html=True)
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("📈 Pontuação por time")
    graf = tabela.sort_values("P", ascending=True)
    fig = px.bar(
        graf, x="P", y="time", orientation="h", text="P",
        labels={"P": "Pontos", "time": ""}, height=600,
    )
    fig.update_traces(marker_color="#1565c0", textposition="outside")
    st.plotly_chart(fig, width="stretch")

# =========================================================================== #
# ABA 2 — Análise de times / features (Fase 2)
# =========================================================================== #
with aba_analise:
    st.subheader("🔬 Features por time (estado atual)")
    st.caption(
        "Indicadores calculados a partir do histórico de jogos — a mesma lógica "
        "que gera o dataset de modelagem da Fase 3."
    )
    st.dataframe(resumo, width="stretch", hide_index=True)

    st.divider()
    st.subheader("⚔️ Comparar dois times")
    c1, c2 = st.columns(2)
    lista = sorted(resumo["time"].tolist())
    time_a = c1.selectbox("Time A", lista, index=0, key="cmp_a")
    time_b = c2.selectbox("Time B", lista, index=1, key="cmp_b")

    metricas = [
        "forma5", "aprov_casa", "aprov_fora",
        "media_gols_pro5", "media_gols_sofr5", "saldo_temporada",
    ]
    ra = resumo[resumo["time"] == time_a][metricas].iloc[0]
    rb = resumo[resumo["time"] == time_b][metricas].iloc[0]
    comp = pd.DataFrame(
        {"métrica": metricas * 2,
         "valor": list(ra.values) + list(rb.values),
         "time": [time_a] * len(metricas) + [time_b] * len(metricas)}
    )
    fig_cmp = px.bar(
        comp, x="métrica", y="valor", color="time", barmode="group",
        labels={"valor": "", "métrica": ""},
    )
    st.plotly_chart(fig_cmp, width="stretch")

    st.divider()
    st.subheader("🏠 Aproveitamento: casa vs fora")
    st.caption("Cada ponto é um time. Acima da diagonal = joga melhor fora do que em casa.")
    fig_sc = px.scatter(
        resumo, x="aprov_casa", y="aprov_fora", text="time",
        labels={"aprov_casa": "Aproveitamento em casa", "aprov_fora": "Aproveitamento fora"},
        height=550,
    )
    fig_sc.update_traces(textposition="top center", marker=dict(size=10, color="#1565c0"))
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig_sc, width="stretch")

    st.divider()
    st.subheader("🔗 Correlação entre as features")
    st.caption(
        "Quão relacionadas as features estão entre si (e com o resultado). "
        "Base para escolher features na Fase 3."
    )
    # alvo numérico só para visualizar correlação com o resultado
    mapa_alvo = {"Fora": -1, "Empate": 0, "Casa": 1}
    num = dataset.select_dtypes("number").copy()
    num["alvo_num"] = dataset["alvo"].map(mapa_alvo)
    corr = num.corr(numeric_only=True).round(2)
    fig_hm = px.imshow(
        corr, text_auto=False, aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, height=700,
    )
    st.plotly_chart(fig_hm, width="stretch")
