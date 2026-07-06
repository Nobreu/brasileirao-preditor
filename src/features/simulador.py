"""Fase 2+ — Simulador de partidas e projeção de temporada.

Modelo estatístico: **Poisson** (o padrão clássico do futebol). A partir dos
gols reais marcados/sofridos por cada time (separando casa e fora), estimamos:

- Gols esperados de um confronto (lambda de cada lado).
- Probabilidade de cada placar -> P(vitória casa / empate / vitória fora),
  ambos marcam (BTTS), over/under 2.5, placar mais provável.
- Projeção do resto do campeonato via Monte Carlo (título, G4, rebaixamento).

⚠️  Assunção: os gols de casa e fora são independentes (simplificação didática;
    a correção de Dixon-Coles para placares baixos fica como refinamento futuro).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from ..data.coletar_tabela import derivar_tabela

SHRINKAGE = 4.0   # "pseudo-jogos" que puxam os ratings para 1.0 (estabiliza início de temporada)
MAX_GOLS = 6      # tamanho da matriz de placares (0..6)


@dataclass
class Forcas:
    """Ratings de ataque/defesa por time + médias da liga."""
    ratings: pd.DataFrame          # index=time, colunas ataque/defesa casa/fora
    media_gols_casa: float         # média de gols do mandante na liga
    media_gols_fora: float         # média de gols do visitante na liga

    @property
    def times(self) -> list[str]:
        return list(self.ratings.index)


# --------------------------------------------------------------------------- #
# 1. Força de cada time (a partir dos jogos finalizados)
# --------------------------------------------------------------------------- #
def _shrink(media: float, n: int, alvo: float) -> float:
    """Puxa uma média para `alvo` quando há poucos jogos (n pequeno)."""
    return (n * media + SHRINKAGE * alvo) / (n + SHRINKAGE) if (n + SHRINKAGE) else alvo


def forcas_times(partidas: pd.DataFrame) -> Forcas:
    media_casa = partidas["gols_casa"].mean()
    media_fora = partidas["gols_fora"].mean()

    linhas = {}
    times = sorted(pd.unique(partidas[["time_casa", "time_fora"]].values.ravel()))
    for t in times:
        jogos_casa = partidas[partidas["time_casa"] == t]
        jogos_fora = partidas[partidas["time_fora"] == t]
        n_casa, n_fora = len(jogos_casa), len(jogos_fora)

        gm_casa = jogos_casa["gols_casa"].mean() if n_casa else media_casa
        gs_casa = jogos_casa["gols_fora"].mean() if n_casa else media_fora
        gm_fora = jogos_fora["gols_fora"].mean() if n_fora else media_fora
        gs_fora = jogos_fora["gols_casa"].mean() if n_fora else media_casa

        linhas[t] = {
            # ratings relativos à média da liga, com shrinkage
            "ataque_casa": _shrink(gm_casa, n_casa, media_casa) / media_casa,
            "defesa_casa": _shrink(gs_casa, n_casa, media_fora) / media_fora,
            "ataque_fora": _shrink(gm_fora, n_fora, media_fora) / media_fora,
            "defesa_fora": _shrink(gs_fora, n_fora, media_casa) / media_casa,
        }
    ratings = pd.DataFrame.from_dict(linhas, orient="index")
    return Forcas(ratings=ratings, media_gols_casa=media_casa, media_gols_fora=media_fora)


# --------------------------------------------------------------------------- #
# 2. Gols esperados e matriz de placares
# --------------------------------------------------------------------------- #
def gols_esperados(forcas: Forcas, casa: str, fora: str) -> tuple[float, float]:
    r = forcas.ratings
    lam_casa = forcas.media_gols_casa * r.at[casa, "ataque_casa"] * r.at[fora, "defesa_fora"]
    lam_fora = forcas.media_gols_fora * r.at[fora, "ataque_fora"] * r.at[casa, "defesa_casa"]
    # piso para evitar lambda zero/degenerado
    return max(lam_casa, 0.05), max(lam_fora, 0.05)


def matriz_placares(lam_casa: float, lam_fora: float, max_gols: int = MAX_GOLS) -> np.ndarray:
    """Matriz (max_gols+1)² com P(placar a x b). Normalizada para somar 1."""
    faixa = np.arange(max_gols + 1)
    ph = poisson.pmf(faixa, lam_casa)
    pa = poisson.pmf(faixa, lam_fora)
    m = np.outer(ph, pa)
    return m / m.sum()   # normaliza (compensa o truncamento em max_gols)


def probabilidades(matriz: np.ndarray) -> dict:
    n = matriz.shape[0]
    idx = np.arange(n)
    casa = np.tril(matriz, -1).sum()    # a > b
    fora = np.triu(matriz, 1).sum()     # a < b
    empate = np.trace(matriz)           # a == b
    btts = matriz[1:, 1:].sum()         # ambos >= 1
    soma_gols = idx[:, None] + idx[None, :]
    over25 = matriz[soma_gols >= 3].sum()

    a, b = np.unravel_index(np.argmax(matriz), matriz.shape)
    # top 5 placares
    planos = [((i, j), matriz[i, j]) for i in range(n) for j in range(n)]
    top5 = sorted(planos, key=lambda x: x[1], reverse=True)[:5]

    return {
        "prob_casa": float(casa),
        "prob_empate": float(empate),
        "prob_fora": float(fora),
        "btts": float(btts),
        "over25": float(over25),
        "placar_mais_provavel": (int(a), int(b)),
        "top5_placares": [((int(i), int(j)), float(p)) for (i, j), p in top5],
    }


def simular_partida(partidas: pd.DataFrame, casa: str, fora: str,
                    forcas: Forcas | None = None) -> dict:
    """Simulação analítica completa de um confronto."""
    forcas = forcas or forcas_times(partidas)
    lam_casa, lam_fora = gols_esperados(forcas, casa, fora)
    matriz = matriz_placares(lam_casa, lam_fora)
    res = probabilidades(matriz)
    res.update({
        "time_casa": casa, "time_fora": fora,
        "lam_casa": float(lam_casa), "lam_fora": float(lam_fora),
        "matriz": matriz,
    })
    return res


# --------------------------------------------------------------------------- #
# 3. Monte Carlo de um confronto (confirma o modelo analítico)
# --------------------------------------------------------------------------- #
def monte_carlo_partida(lam_casa: float, lam_fora: float,
                        n: int = 10000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    gc = rng.poisson(lam_casa, n)
    gf = rng.poisson(lam_fora, n)
    return {
        "n": n,
        "freq_casa": float(np.mean(gc > gf)),
        "freq_empate": float(np.mean(gc == gf)),
        "freq_fora": float(np.mean(gc < gf)),
        "media_gols_casa": float(gc.mean()),
        "media_gols_fora": float(gf.mean()),
        "total_gols": (gc + gf),   # array, para histograma
    }


# --------------------------------------------------------------------------- #
# 4. Narração automática
# --------------------------------------------------------------------------- #
def narrar(res: dict) -> str:
    casa, fora = res["time_casa"], res["time_fora"]
    pc, pe, pf = res["prob_casa"], res["prob_empate"], res["prob_fora"]
    a, b = res["placar_mais_provavel"]
    maior = max(pc, pe, pf)
    if maior == pc:
        favorito = f"O **{casa}** entra como favorito jogando em casa"
    elif maior == pf:
        favorito = f"O **{fora}** é favorito mesmo atuando fora"
    else:
        favorito = "O equilíbrio é grande — o empate é o resultado mais provável"

    if abs(pc - pf) < 0.10:
        margem = "mas é um jogo muito parelho"
    elif abs(pc - pf) < 0.25:
        margem = "com uma vantagem moderada"
    else:
        margem = "com folga clara no papel"

    total = res["lam_casa"] + res["lam_fora"]
    tendencia = "jogo aberto, com gols" if total >= 2.6 else "jogo mais truncado, de poucos gols"

    return (
        f"{favorito} ({maior:.0%}), {margem}. "
        f"A expectativa é de {res['lam_casa']:.1f} a {res['lam_fora']:.1f} em gols esperados "
        f"({tendencia}). O placar mais provável é **{a}x{b}**, "
        f"com {res['over25']:.0%} de chance de sair mais de 2,5 gols e "
        f"{res['btts']:.0%} de ambos marcarem."
    )


# --------------------------------------------------------------------------- #
# 5. Projeção de temporada (Monte Carlo do resto do campeonato)
# --------------------------------------------------------------------------- #
def projetar_temporada(partidas: pd.DataFrame, jogos_futuros: pd.DataFrame,
                       forcas: Forcas | None = None,
                       n_sims: int = 10000, seed: int = 42,
                       n_rebaixados: int = 4, n_g4: int = 4) -> pd.DataFrame:
    """Projeta o campeonato final simulando os jogos que faltam.

    Devolve, por time: prob. de título, G4 e rebaixamento, além de pontos e
    posição média projetados. Vetorizado em numpy sobre `n_sims` simulações.
    """
    tabela = derivar_tabela(partidas)
    times = sorted(tabela["time"].tolist())
    idx = {t: i for i, t in enumerate(times)}
    T = len(times)

    pontos0 = np.array([tabela.set_index("time").at[t, "P"] for t in times], dtype=float)
    saldo0 = np.array([tabela.set_index("time").at[t, "SG"] for t in times], dtype=float)

    if jogos_futuros is None or jogos_futuros.empty:
        return pd.DataFrame()

    # só jogos entre times conhecidos da tabela
    fut = jogos_futuros[
        jogos_futuros["time_casa"].isin(idx) & jogos_futuros["time_fora"].isin(idx)
    ]
    if fut.empty:
        return pd.DataFrame()

    forcas = forcas or forcas_times(partidas)
    rng = np.random.default_rng(seed)

    pontos = np.tile(pontos0, (n_sims, 1))    # (n_sims, T)
    saldo = np.tile(saldo0, (n_sims, 1))

    for _, jogo in fut.iterrows():
        ic, ifr = idx[jogo["time_casa"]], idx[jogo["time_fora"]]
        lc, lf = gols_esperados(forcas, jogo["time_casa"], jogo["time_fora"])
        gc = rng.poisson(lc, n_sims)
        gf = rng.poisson(lf, n_sims)
        casa_venceu = gc > gf
        fora_venceu = gc < gf
        empate = gc == gf
        pontos[:, ic] += np.where(casa_venceu, 3, np.where(empate, 1, 0))
        pontos[:, ifr] += np.where(fora_venceu, 3, np.where(empate, 1, 0))
        saldo[:, ic] += gc - gf
        saldo[:, ifr] += gf - gc

    # chave de classificação: pontos com desempate por saldo
    chave = pontos + saldo / 1000.0
    # posição (1 = melhor) via duplo argsort
    posicoes = (-chave).argsort(axis=1).argsort(axis=1) + 1

    campeao = posicoes == 1
    g4 = posicoes <= n_g4
    z4 = posicoes >= (T - n_rebaixados + 1)

    resumo = pd.DataFrame({
        "time": times,
        "pontos_atuais": pontos0.astype(int),
        "pontos_proj_medio": pontos.mean(axis=0).round(1),
        "posicao_media": posicoes.mean(axis=0).round(1),
        "prob_titulo": campeao.mean(axis=0),
        "prob_g4": g4.mean(axis=0),
        "prob_rebaixamento": z4.mean(axis=0),
    })
    return resumo.sort_values(
        ["prob_titulo", "pontos_proj_medio"], ascending=False
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Cenário mais provável — palpite jogo a jogo e tabela final montada
# --------------------------------------------------------------------------- #
def prever_jogos_futuros(partidas: pd.DataFrame, jogos_futuros: pd.DataFrame,
                         forcas: Forcas | None = None) -> pd.DataFrame:
    """Para cada jogo futuro, o placar mais provável + probabilidades 1/X/2.

    É o palpite jogo a jogo (um cenário, o mais provável de cada partida).
    """
    if jogos_futuros is None or jogos_futuros.empty:
        return pd.DataFrame()
    forcas = forcas or forcas_times(partidas)
    linhas = []
    for _, j in jogos_futuros.iterrows():
        casa, fora = j["time_casa"], j["time_fora"]
        if casa not in forcas.ratings.index or fora not in forcas.ratings.index:
            continue
        r = simular_partida(partidas, casa, fora, forcas=forcas)
        a, b = r["placar_mais_provavel"]
        tendencia = max(
            (("Casa", r["prob_casa"]), ("Empate", r["prob_empate"]), ("Fora", r["prob_fora"])),
            key=lambda x: x[1],
        )[0]
        linhas.append({
            "rodada": int(j["rodada"]) if pd.notna(j.get("rodada")) else None,
            "data": j.get("data"),
            "time_casa": casa,
            "time_fora": fora,
            "gols_casa": a,
            "gols_fora": b,
            "placar_provavel": f"{a} x {b}",
            "prob_casa": r["prob_casa"],
            "prob_empate": r["prob_empate"],
            "prob_fora": r["prob_fora"],
            "tendencia": tendencia,
        })
    return pd.DataFrame(linhas)


def tabela_projetada(partidas: pd.DataFrame, jogos_futuros: pd.DataFrame,
                     forcas: Forcas | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monta a tabela final assumindo o placar mais provável de cada jogo restante.

    Devolve `(tabela_final, palpites_por_jogo)`. A tabela final considera TODOS
    os jogos (finalizados + os futuros no cenário mais provável).
    """
    forcas = forcas or forcas_times(partidas)
    palpites = prever_jogos_futuros(partidas, jogos_futuros, forcas=forcas)

    cols = ["rodada", "time_casa", "time_fora", "gols_casa", "gols_fora"]
    reais = partidas[cols]
    if palpites.empty:
        combinado = reais
    else:
        combinado = pd.concat([reais, palpites[cols]], ignore_index=True)
    tabela_final = derivar_tabela(combinado)
    return tabela_final, palpites
