"""Cálculo da forma recente (últimas N partidas de cada time).

Para cada time, olha os últimos N jogos (em ordem cronológica) e resume:
- sequência de resultados (V/E/D), do mais antigo ao mais recente
- pontos conquistados nesses jogos (V=3, E=1, D=0)
- gols marcados e sofridos
"""
from __future__ import annotations

import pandas as pd

from .paths import FORMA_PROC, PARTIDAS_PROC, garantir_pastas


def _jogos_do_time(partidas: pd.DataFrame, time: str) -> pd.DataFrame:
    """Devolve os jogos do time na perspectiva dele (marcou/sofreu/resultado)."""
    em_casa = partidas[partidas["time_casa"] == time].copy()
    em_casa["adversario"] = em_casa["time_fora"]
    em_casa["gols_pro"] = em_casa["gols_casa"]
    em_casa["gols_contra"] = em_casa["gols_fora"]

    fora = partidas[partidas["time_fora"] == time].copy()
    fora["adversario"] = fora["time_casa"]
    fora["gols_pro"] = fora["gols_fora"]
    fora["gols_contra"] = fora["gols_casa"]

    jogos = pd.concat([em_casa, fora]).sort_values(["rodada", "data"])
    jogos["res"] = jogos.apply(
        lambda r: "V" if r.gols_pro > r.gols_contra
        else ("D" if r.gols_pro < r.gols_contra else "E"),
        axis=1,
    )
    return jogos


def calcular_forma(partidas: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    times = sorted(
        pd.unique(partidas[["time_casa", "time_fora"]].values.ravel())
    )
    pontos_por_res = {"V": 3, "E": 1, "D": 0}
    linhas = []
    for time in times:
        ultimos = _jogos_do_time(partidas, time).tail(n)
        seq = list(ultimos["res"])
        linhas.append(
            {
                "time": time,
                "sequencia": "-".join(seq),  # antigo -> recente
                "pontos_ultimos_n": int(sum(pontos_por_res[r] for r in seq)),
                "vitorias": seq.count("V"),
                "empates": seq.count("E"),
                "derrotas": seq.count("D"),
                "gols_pro": int(ultimos["gols_pro"].sum()),
                "gols_contra": int(ultimos["gols_contra"].sum()),
                "n_jogos": len(seq),
            }
        )
    forma = pd.DataFrame(linhas).sort_values(
        ["pontos_ultimos_n", "gols_pro"], ascending=False
    ).reset_index(drop=True)
    return forma


def main(n: int = 5) -> None:
    garantir_pastas()
    partidas = pd.read_csv(PARTIDAS_PROC)
    forma = calcular_forma(partidas, n=n)
    forma.to_csv(FORMA_PROC, index=False, encoding="utf-8")
    print(f"[forma] forma recente (últimas {n}) -> {FORMA_PROC}")


if __name__ == "__main__":
    main()
