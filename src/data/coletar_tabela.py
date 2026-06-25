"""Coleta os dados do Brasileirão.

Estratégia da Fase 1:
1. Se houver uma chave de API (variável de ambiente FOOTBALL_DATA_API_KEY),
   tenta buscar dados reais em football-data.org.
2. Se não houver chave OU a chamada falhar, gera uma temporada SIMULADA
   determinística (mesma semente -> mesmos dados). Assim a aula sempre roda,
   mesmo sem internet.

Em ambos os casos salvamos:
    data/raw/partidas.csv      -> uma linha por jogo
    data/raw/tabela_atual.csv  -> classificação derivada dos jogos
"""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .paths import PARTIDAS_RAW, TABELA_RAW, garantir_pastas
from .times import FORCA, TIMES_SERIE_A, normalizar_nome

SEMENTE = 42
DATA_INICIO = date(2024, 4, 13)  # início aproximado do Brasileirão 2024


# --------------------------------------------------------------------------- #
# 1. Geração do calendário (turno e returno) — método do círculo
# --------------------------------------------------------------------------- #
def _gerar_calendario(times: list[str]) -> list[tuple[int, str, str]]:
    """Round-robin duplo: cada par se enfrenta em casa e fora.

    Devolve lista de (rodada, time_casa, time_fora).
    """
    n = len(times)
    rodada_meia = n - 1  # 19 rodadas no turno
    fixos = times[0]
    rotativos = times[1:]
    jogos: list[tuple[int, str, str]] = []

    for r in range(rodada_meia):
        ordem = [fixos] + rotativos
        for i in range(n // 2):
            casa = ordem[i]
            fora = ordem[n - 1 - i]
            # alterna o mando a cada rodada para distribuir jogos em casa
            if r % 2 == 0:
                jogos.append((r + 1, casa, fora))
            else:
                jogos.append((r + 1, fora, casa))
        # rotaciona mantendo o primeiro fixo
        rotativos = [rotativos[-1]] + rotativos[:-1]

    # returno: mesmos confrontos com mando invertido, rodadas 20..38
    returno = [(r + rodada_meia, fora, casa) for (r, casa, fora) in jogos]
    return jogos + returno


# --------------------------------------------------------------------------- #
# 2. Simulação dos placares
# --------------------------------------------------------------------------- #
def simular_temporada(semente: int = SEMENTE) -> pd.DataFrame:
    """Gera placares plausíveis usando a força de cada time (Poisson)."""
    rng = np.random.default_rng(semente)
    calendario = _gerar_calendario(TIMES_SERIE_A)

    linhas = []
    for rodada, casa, fora in calendario:
        f_casa = FORCA[casa]
        f_fora = FORCA[fora]
        # gols esperados: força do ataque relativa à defesa + vantagem de casa
        lambda_casa = 1.35 * (f_casa / f_fora) ** 0.9
        lambda_fora = 1.05 * (f_fora / f_casa) ** 0.9
        gols_casa = int(rng.poisson(lambda_casa))
        gols_fora = int(rng.poisson(lambda_fora))
        data_jogo = DATA_INICIO + timedelta(days=(rodada - 1) * 7)
        linhas.append(
            {
                "rodada": rodada,
                "data": data_jogo.isoformat(),
                "time_casa": casa,
                "time_fora": fora,
                "gols_casa": gols_casa,
                "gols_fora": gols_fora,
            }
        )

    df = pd.DataFrame(linhas).sort_values(["rodada", "time_casa"]).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. (Opcional) Coleta real via football-data.org
# --------------------------------------------------------------------------- #
def coletar_api() -> pd.DataFrame | None:
    """Tenta baixar partidas reais. Devolve None se não der certo."""
    chave = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not chave:
        return None
    try:
        import requests

        url = "https://api.football-data.org/v4/competitions/BSA/matches"
        resp = requests.get(url, headers={"X-Auth-Token": chave}, timeout=15)
        resp.raise_for_status()
        dados = resp.json()
        linhas = []
        for m in dados.get("matches", []):
            if m.get("status") != "FINISHED":
                continue
            placar = m["score"]["fullTime"]
            linhas.append(
                {
                    "rodada": m.get("matchday"),
                    "data": m["utcDate"][:10],
                    "time_casa": m["homeTeam"]["name"],
                    "time_fora": m["awayTeam"]["name"],
                    "gols_casa": placar["home"],
                    "gols_fora": placar["away"],
                }
            )
        if not linhas:
            return None
        return pd.DataFrame(linhas)
    except Exception as exc:  # rede instável, limite de API, etc.
        print(f"[coletar_api] Falhou, usando simulação. Motivo: {exc}")
        return None


# --------------------------------------------------------------------------- #
# 4. Tabela derivada das partidas
# --------------------------------------------------------------------------- #
def derivar_tabela(partidas: pd.DataFrame) -> pd.DataFrame:
    """Calcula a classificação a partir dos jogos finalizados."""
    estat = {
        t: {"P": 0, "J": 0, "V": 0, "E": 0, "D": 0, "GP": 0, "GC": 0}
        for t in pd.unique(partidas[["time_casa", "time_fora"]].values.ravel())
    }

    for _, jogo in partidas.iterrows():
        casa, fora = jogo["time_casa"], jogo["time_fora"]
        gc, gf = int(jogo["gols_casa"]), int(jogo["gols_fora"])
        for t, gp, gs in ((casa, gc, gf), (fora, gf, gc)):
            estat[t]["J"] += 1
            estat[t]["GP"] += gp
            estat[t]["GC"] += gs
        if gc > gf:
            estat[casa]["V"] += 1
            estat[casa]["P"] += 3
            estat[fora]["D"] += 1
        elif gc < gf:
            estat[fora]["V"] += 1
            estat[fora]["P"] += 3
            estat[casa]["D"] += 1
        else:
            estat[casa]["E"] += 1
            estat[fora]["E"] += 1
            estat[casa]["P"] += 1
            estat[fora]["P"] += 1

    tabela = pd.DataFrame(
        [{"time": t, **v} for t, v in estat.items()]
    )
    tabela["SG"] = tabela["GP"] - tabela["GC"]
    tabela = tabela.sort_values(
        ["P", "V", "SG", "GP"], ascending=False
    ).reset_index(drop=True)
    tabela.insert(0, "posicao", tabela.index + 1)
    return tabela


# --------------------------------------------------------------------------- #
# 5. Orquestração
# --------------------------------------------------------------------------- #
def main() -> None:
    garantir_pastas()
    partidas = coletar_api()
    origem = "API football-data.org"
    if partidas is None:
        partidas = simular_temporada()
        origem = "simulação determinística"

    # padroniza nomes já na coleta (a limpeza reforça isso depois)
    partidas["time_casa"] = partidas["time_casa"].map(normalizar_nome)
    partidas["time_fora"] = partidas["time_fora"].map(normalizar_nome)

    tabela = derivar_tabela(partidas)

    partidas.to_csv(PARTIDAS_RAW, index=False, encoding="utf-8")
    tabela.to_csv(TABELA_RAW, index=False, encoding="utf-8")
    print(f"[coletar_tabela] Fonte: {origem}")
    print(f"[coletar_tabela] {len(partidas)} partidas -> {PARTIDAS_RAW}")
    print(f"[coletar_tabela] {len(tabela)} times    -> {TABELA_RAW}")


if __name__ == "__main__":
    main()
