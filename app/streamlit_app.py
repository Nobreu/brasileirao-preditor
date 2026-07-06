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
    JOGOS_FUTUROS_PROC,
    PARTIDAS_PROC,
    TABELA_PROC,
)
from src.data import preparar_dados  # noqa: E402
from src.features.build_features import resumo_times  # noqa: E402
from src.features.simulador import (  # noqa: E402
    forcas_times,
    monte_carlo_partida,
    narrar,
    projetar_temporada,
    simular_partida,
)

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
    if JOGOS_FUTUROS_PROC.exists():
        jogos_futuros = pd.read_csv(JOGOS_FUTUROS_PROC)
    else:
        jogos_futuros = pd.DataFrame(columns=["rodada", "data", "time_casa", "time_fora"])
    forcas = forcas_times(partidas)
    return tabela, forma, partidas, dataset, resumo, jogos_futuros, forcas


@st.cache_data
def carregar_projecao(n_sims: int):
    partidas = pd.read_csv(PARTIDAS_PROC)
    futuros = pd.read_csv(JOGOS_FUTUROS_PROC)
    return projetar_temporada(partidas, futuros, n_sims=n_sims)


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
tabela, forma, partidas, dataset, resumo, jogos_futuros, forcas = carregar_dados()

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

aba_tabela, aba_analise, aba_sim, aba_proj = st.tabs(
    ["📊 Classificação & Forma", "🔬 Análise de Times", "🎲 Simulador", "🏆 Projeção"]
)

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

# =========================================================================== #
# ABA 3 — Simulador de partida (modelo de Poisson)
# =========================================================================== #
with aba_sim:
    st.subheader("🎲 Simulador de partida")
    st.caption(
        "Modelo de Poisson: estima o resultado de um confronto a partir dos gols "
        "reais marcados/sofridos por cada time (com mando de campo)."
    )
    lista = sorted(tabela["time"].tolist())
    c1, c2 = st.columns(2)
    casa = c1.selectbox("🏠 Mandante", lista, index=0, key="sim_casa")
    fora = c2.selectbox("✈️ Visitante", lista, index=1, key="sim_fora")

    if casa == fora:
        st.warning("Escolha dois times diferentes.")
    else:
        res = simular_partida(partidas, casa, fora, forcas=forcas)

        m1, m2, m3 = st.columns(3)
        m1.metric(f"Vitória {casa}", f"{res['prob_casa']:.0%}")
        m2.metric("Empate", f"{res['prob_empate']:.0%}")
        m3.metric(f"Vitória {fora}", f"{res['prob_fora']:.0%}")

        a, b = res["placar_mais_provavel"]
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Gols esperados", f"{res['lam_casa']:.1f} x {res['lam_fora']:.1f}")
        g2.metric("Placar provável", f"{a} x {b}")
        g3.metric("Ambos marcam", f"{res['btts']:.0%}")
        g4.metric("Mais de 2,5 gols", f"{res['over25']:.0%}")

        st.info(narrar(res))

        col_hm, col_top = st.columns([2, 1])
        with col_hm:
            st.markdown("**Distribuição de placares** (% de probabilidade)")
            faixa = list(range(res["matriz"].shape[0]))
            fig_pl = px.imshow(
                res["matriz"] * 100,
                x=faixa, y=faixa, text_auto=".1f",
                labels={"x": f"Gols {fora}", "y": f"Gols {casa}", "color": "%"},
                color_continuous_scale="Blues", aspect="auto", height=430,
            )
            st.plotly_chart(fig_pl, width="stretch")
        with col_top:
            st.markdown("**Placares mais prováveis**")
            for (i, j), prob in res["top5_placares"]:
                st.markdown(f"- **{i} x {j}** — {prob:.1%}")

        with st.expander("🎰 Rodar 10.000 simulações (Monte Carlo)"):
            mc = monte_carlo_partida(res["lam_casa"], res["lam_fora"], n=10000)
            d1, d2, d3 = st.columns(3)
            d1.metric(f"Vitória {casa}", f"{mc['freq_casa']:.1%}")
            d2.metric("Empate", f"{mc['freq_empate']:.1%}")
            d3.metric(f"Vitória {fora}", f"{mc['freq_fora']:.1%}")
            st.caption(
                "As frequências das 10 mil simulações confirmam as probabilidades "
                "do cálculo analítico acima."
            )
            fig_h = px.histogram(
                x=mc["total_gols"],
                labels={"x": "Total de gols na partida"},
                nbins=int(mc["total_gols"].max()) + 1, height=320,
            )
            fig_h.update_traces(marker_color="#1565c0")
            st.plotly_chart(fig_h, width="stretch")

# =========================================================================== #
# ABA 4 — Projeção da temporada (Monte Carlo do resto do campeonato)
# =========================================================================== #
with aba_proj:
    st.subheader("🏆 Projeção da temporada")
    if jogos_futuros.empty:
        st.info(
            "Não há jogos futuros nesta base — a temporada está completa ou os "
            "dados vêm da simulação offline. A projeção precisa de partidas ainda "
            "não disputadas."
        )
        st.caption(
            "Configure o token da API (.env) e rode `python -m src.data.preparar_dados` "
            "durante a temporada para habilitar esta aba."
        )
    else:
        st.caption(
            f"{len(jogos_futuros)} jogos restantes · simulamos o resto do campeonato "
            "milhares de vezes para estimar as chances de cada time."
        )
        n_sims = st.select_slider(
            "Número de simulações", options=[1000, 5000, 10000], value=5000
        )
        with st.spinner("Rodando simulações..."):
            proj = carregar_projecao(n_sims)

        disp = proj.copy()
        for c in ("prob_titulo", "prob_g4", "prob_rebaixamento"):
            disp[c] = (disp[c] * 100).round(1)
        disp = disp.rename(
            columns={
                "time": "Time", "pontos_atuais": "Pts atuais",
                "pontos_proj_medio": "Pts projetados", "posicao_media": "Pos. média",
                "prob_titulo": "Título %", "prob_g4": "G4 %",
                "prob_rebaixamento": "Rebaix. %",
            }
        )
        st.dataframe(disp, width="stretch", hide_index=True)

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**🥇 Chance de título**")
            top = proj[proj["prob_titulo"] > 0].head(8).sort_values("prob_titulo")
            fig_t = px.bar(
                top, x="prob_titulo", y="time", orientation="h",
                labels={"prob_titulo": "", "time": ""},
            )
            fig_t.update_traces(marker_color="#2e7d32")
            fig_t.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_t, width="stretch")
        with cB:
            st.markdown("**🔻 Risco de rebaixamento**")
            riz = proj.sort_values("prob_rebaixamento", ascending=False).head(8)
            riz = riz.sort_values("prob_rebaixamento")
            fig_r = px.bar(
                riz, x="prob_rebaixamento", y="time", orientation="h",
                labels={"prob_rebaixamento": "", "time": ""},
            )
            fig_r.update_traces(marker_color="#c62828")
            fig_r.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig_r, width="stretch")
