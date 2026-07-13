"""Fase 2 — Engenharia de features.

Transforma a lista de partidas em um dataset de MODELAGEM: uma linha por
partida, descrevendo o estado de cada time ANTES do jogo, mais o resultado
(alvo). Esse dataset é o que vai alimentar o modelo na Fase 3.

⚠️  REGRA DE OURO — SEM DATA LEAK (vazamento de dados):
    As features de uma partida só podem usar informação de jogos ANTERIORES
    a ela. Para garantir isso, percorremos os jogos em ordem cronológica e
    calculamos as features a partir do histórico acumulado ATÉ o jogo
    anterior — só depois registramos o resultado do jogo atual no histórico.
"""
from __future__ import annotations

import pandas as pd

from ..data.paths import DATASET_MODELAGEM, PARTIDAS_PROC, garantir_pastas

PONTOS = {"V": 3, "E": 1, "D": 0}
JANELA = 5          # tamanho da "forma recente"
MIN_JOGOS = 3       # só gera linha se ambos os times já têm este nº de jogos


# --------------------------------------------------------------------------- #
# Features de um único time, a partir do seu histórico (lista de jogos passados)
# --------------------------------------------------------------------------- #
def _features_time(hist: list[dict], mando: str) -> dict:
    """Calcula as features de um time dado seu histórico de jogos ANTERIORES.

    `hist` é uma lista de dicts, cada um com: res, pontos, gp (gols pró),
    gc (gols contra), casa (bool). Ordenada do mais antigo ao mais recente.
    `mando` é "casa" ou "fora": indica como o time entra NESTE jogo.
    """
    ultimos = hist[-JANELA:]                       # os JANELA jogos mais recentes
    n = len(hist)

    # forma recente: pontos nos últimos N jogos
    forma = sum(j["pontos"] for j in ultimos)

    # médias de gols nos últimos N
    media_gp = sum(j["gp"] for j in ultimos) / len(ultimos)
    media_gc = sum(j["gc"] for j in ultimos) / len(ultimos)

    # saldo de gols na temporada (todos os jogos anteriores)
    saldo = sum(j["gp"] - j["gc"] for j in hist)

    # aproveitamento geral (pontos ganhos / pontos possíveis)
    aprov_geral = sum(j["pontos"] for j in hist) / (3 * n)

    # aproveitamento no mando específico deste jogo (só jogos no mesmo mando)
    quer_casa = mando == "casa"
    mesmos = [j for j in hist if j["casa"] == quer_casa]
    if mesmos:
        aprov_mando = sum(j["pontos"] for j in mesmos) / (3 * len(mesmos))
    else:
        aprov_mando = aprov_geral                  # sem histórico no mando: usa o geral

    # sequência atual sem perder / sem vencer (conta do fim para trás)
    sem_derrota = 0
    for j in reversed(hist):
        if j["res"] == "D":
            break
        sem_derrota += 1
    sem_vitoria = 0
    for j in reversed(hist):
        if j["res"] == "V":
            break
        sem_vitoria += 1

    return {
        "forma5": forma,
        "media_gols_pro5": round(media_gp, 3),
        "media_gols_sofr5": round(media_gc, 3),
        "saldo_temporada": saldo,
        "aprov_geral": round(aprov_geral, 3),
        "aprov_mando": round(aprov_mando, 3),
        "sem_derrota": sem_derrota,
        "sem_vitoria": sem_vitoria,
    }


def _registrar(hist: dict, time: str, gp: int, gc: int, casa: bool) -> None:
    """Acrescenta um jogo ao histórico de um time (chamado DEPOIS de gerar features)."""
    res = "V" if gp > gc else ("D" if gp < gc else "E")
    hist.setdefault(time, []).append(
        {"res": res, "pontos": PONTOS[res], "gp": gp, "gc": gc, "casa": casa}
    )


# --------------------------------------------------------------------------- #
# Dataset de modelagem: uma linha por partida
# --------------------------------------------------------------------------- #
def construir_dataset(
    partidas: pd.DataFrame, janela: int = JANELA, min_jogos: int = MIN_JOGOS
) -> pd.DataFrame:
    """Gera o dataset de modelagem sem data leak."""
    df = partidas.copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values(["rodada", "data"]).reset_index(drop=True)

    hist: dict[str, list[dict]] = {}          # histórico por time
    h2h: dict[frozenset, list[str]] = {}      # confrontos diretos (vencedor por jogo)
    linhas = []

    for _, jogo in df.iterrows():
        casa, fora = jogo["time_casa"], jogo["time_fora"]
        gc, gf = int(jogo["gols_casa"]), int(jogo["gols_fora"])

        h_casa = hist.get(casa, [])
        h_fora = hist.get(fora, [])

        # só gera a linha se ambos já têm histórico suficiente (evita ruído)
        if len(h_casa) >= min_jogos and len(h_fora) >= min_jogos:
            fc = _features_time(h_casa, "casa")
            ff = _features_time(h_fora, "fora")

            # confronto direto: saldo de vitórias do mandante nos jogos passados
            par = frozenset((casa, fora))
            anteriores = h2h.get(par, [])
            h2h_saldo = anteriores.count(casa) - anteriores.count(fora)

            # alvo: quem venceu (na perspectiva do mando)
            alvo = "Casa" if gc > gf else ("Fora" if gc < gf else "Empate")

            linhas.append(
                {
                    "rodada": int(jogo["rodada"]),
                    "data": jogo["data"].date() if pd.notna(jogo["data"]) else None,
                    "time_casa": casa,
                    "time_fora": fora,
                    # features do mandante (prefixo casa_)
                    "casa_forma5": fc["forma5"],
                    "casa_aprov_mando": fc["aprov_mando"],
                    "casa_media_gols_pro5": fc["media_gols_pro5"],
                    "casa_media_gols_sofr5": fc["media_gols_sofr5"],
                    "casa_saldo_temporada": fc["saldo_temporada"],
                    "casa_sem_derrota": fc["sem_derrota"],
                    "casa_sem_vitoria": fc["sem_vitoria"],
                    # features do visitante (prefixo fora_)
                    "fora_forma5": ff["forma5"],
                    "fora_aprov_mando": ff["aprov_mando"],
                    "fora_media_gols_pro5": ff["media_gols_pro5"],
                    "fora_media_gols_sofr5": ff["media_gols_sofr5"],
                    "fora_saldo_temporada": ff["saldo_temporada"],
                    "fora_sem_derrota": ff["sem_derrota"],
                    "fora_sem_vitoria": ff["sem_vitoria"],
                    # comparativas (diferença mandante - visitante)
                    "diff_forma5": fc["forma5"] - ff["forma5"],
                    "diff_saldo": fc["saldo_temporada"] - ff["saldo_temporada"],
                    "diff_aprov_mando": round(fc["aprov_mando"] - ff["aprov_mando"], 3),
                    # confronto direto
                    "h2h_saldo_casa": h2h_saldo,
                    # alvo
                    "alvo": alvo,
                }
            )

        # AGORA (e só agora) registramos o jogo atual no histórico
        _registrar(hist, casa, gc, gf, casa=True)
        _registrar(hist, fora, gf, gc, casa=False)
        vencedor = casa if gc > gf else (fora if gc < gf else "Empate")
        h2h.setdefault(frozenset((casa, fora)), []).append(vencedor)

    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
# Resumo do estado ATUAL de cada time (para o dashboard) — descritivo, sem alvo
# --------------------------------------------------------------------------- #
def resumo_times(partidas: pd.DataFrame, janela: int = JANELA) -> pd.DataFrame:
    """Estado atual de cada time considerando TODOS os jogos disputados."""
    df = partidas.copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values(["rodada", "data"]).reset_index(drop=True)

    hist: dict[str, list[dict]] = {}
    for _, jogo in df.iterrows():
        casa, fora = jogo["time_casa"], jogo["time_fora"]
        gc, gf = int(jogo["gols_casa"]), int(jogo["gols_fora"])
        _registrar(hist, casa, gc, gf, casa=True)
        _registrar(hist, fora, gf, gc, casa=False)

    linhas = []
    for time, h in hist.items():
        # aproveitamento em casa e fora separados
        casa_j = [j for j in h if j["casa"]]
        fora_j = [j for j in h if not j["casa"]]
        aprov_casa = sum(j["pontos"] for j in casa_j) / (3 * len(casa_j)) if casa_j else 0
        aprov_fora = sum(j["pontos"] for j in fora_j) / (3 * len(fora_j)) if fora_j else 0
        ult = h[-janela:]
        linhas.append(
            {
                "time": time,
                "jogos": len(h),
                "forma5": sum(j["pontos"] for j in ult),
                "aprov_casa": round(aprov_casa, 3),
                "aprov_fora": round(aprov_fora, 3),
                "media_gols_pro5": round(sum(j["gp"] for j in ult) / len(ult), 2),
                "media_gols_sofr5": round(sum(j["gc"] for j in ult) / len(ult), 2),
                "saldo_temporada": sum(j["gp"] - j["gc"] for j in h),
            }
        )
    return pd.DataFrame(linhas).sort_values("saldo_temporada", ascending=False).reset_index(drop=True)


# Colunas de feature do dataset (fonte única da verdade para treino e previsão)
FEATURES = [
    "casa_forma5", "casa_aprov_mando", "casa_media_gols_pro5", "casa_media_gols_sofr5",
    "casa_saldo_temporada", "casa_sem_derrota", "casa_sem_vitoria",
    "fora_forma5", "fora_aprov_mando", "fora_media_gols_pro5", "fora_media_gols_sofr5",
    "fora_saldo_temporada", "fora_sem_derrota", "fora_sem_vitoria",
    "diff_forma5", "diff_saldo", "diff_aprov_mando", "h2h_saldo_casa",
]


def montar_features_atual(partidas: pd.DataFrame, casa: str, fora: str) -> dict | None:
    """Monta as features de um confronto HIPOTÉTICO com o estado ATUAL dos times.

    Usa TODO o histórico de cada time (não há partida futura para vazar), gerando
    exatamente as mesmas colunas que `construir_dataset` — garantindo que o modelo
    receba na previsão o mesmo formato que viu no treino. Devolve None se algum
    dos times não tiver histórico.
    """
    df = partidas.copy()
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df.sort_values(["rodada", "data"]).reset_index(drop=True)

    hist: dict[str, list[dict]] = {}
    h2h: dict[frozenset, list[str]] = {}
    for _, jogo in df.iterrows():
        c, f = jogo["time_casa"], jogo["time_fora"]
        gc, gf = int(jogo["gols_casa"]), int(jogo["gols_fora"])
        _registrar(hist, c, gc, gf, casa=True)
        _registrar(hist, f, gf, gc, casa=False)
        vencedor = c if gc > gf else (f if gc < gf else "Empate")
        h2h.setdefault(frozenset((c, f)), []).append(vencedor)

    h_casa, h_fora = hist.get(casa), hist.get(fora)
    if not h_casa or not h_fora:
        return None

    fc = _features_time(h_casa, "casa")
    ff = _features_time(h_fora, "fora")
    anteriores = h2h.get(frozenset((casa, fora)), [])
    return {
        "casa_forma5": fc["forma5"],
        "casa_aprov_mando": fc["aprov_mando"],
        "casa_media_gols_pro5": fc["media_gols_pro5"],
        "casa_media_gols_sofr5": fc["media_gols_sofr5"],
        "casa_saldo_temporada": fc["saldo_temporada"],
        "casa_sem_derrota": fc["sem_derrota"],
        "casa_sem_vitoria": fc["sem_vitoria"],
        "fora_forma5": ff["forma5"],
        "fora_aprov_mando": ff["aprov_mando"],
        "fora_media_gols_pro5": ff["media_gols_pro5"],
        "fora_media_gols_sofr5": ff["media_gols_sofr5"],
        "fora_saldo_temporada": ff["saldo_temporada"],
        "fora_sem_derrota": ff["sem_derrota"],
        "fora_sem_vitoria": ff["sem_vitoria"],
        "diff_forma5": fc["forma5"] - ff["forma5"],
        "diff_saldo": fc["saldo_temporada"] - ff["saldo_temporada"],
        "diff_aprov_mando": round(fc["aprov_mando"] - ff["aprov_mando"], 3),
        "h2h_saldo_casa": anteriores.count(casa) - anteriores.count(fora),
    }


def main() -> None:
    garantir_pastas()
    partidas = pd.read_csv(PARTIDAS_PROC)
    dataset = construir_dataset(partidas)
    dataset.to_csv(DATASET_MODELAGEM, index=False, encoding="utf-8")
    n_features = len([c for c in dataset.columns
                      if c not in ("rodada", "data", "time_casa", "time_fora", "alvo")])
    print(f"[build_features] {len(dataset)} partidas x {n_features} features -> {DATASET_MODELAGEM}")
    print(f"[build_features] distribuição do alvo:\n{dataset['alvo'].value_counts()}")


if __name__ == "__main__":
    main()
