"""Tratamento dos dados brutos.

Lê data/raw, padroniza nomes de times, converte datas, trata nulos e
salva versões limpas em data/processed.
"""
from __future__ import annotations

import pandas as pd

from .paths import (
    JOGOS_FUTUROS_PROC,
    JOGOS_FUTUROS_RAW,
    PARTIDAS_PROC,
    PARTIDAS_RAW,
    TABELA_PROC,
    garantir_pastas,
)
from .coletar_tabela import derivar_tabela
from .times import normalizar_nome


def limpar_partidas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # padroniza nomes
    df["time_casa"] = df["time_casa"].map(normalizar_nome)
    df["time_fora"] = df["time_fora"].map(normalizar_nome)
    # converte datas
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    # tipos numéricos
    for col in ("rodada", "gols_casa", "gols_fora"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # remove jogos sem placar (não finalizados / corrompidos)
    antes = len(df)
    df = df.dropna(subset=["gols_casa", "gols_fora", "time_casa", "time_fora"])
    df[["gols_casa", "gols_fora"]] = df[["gols_casa", "gols_fora"]].astype(int)
    removidos = antes - len(df)
    if removidos:
        print(f"[limpar] {removidos} partidas sem placar removidas")
    # resultado em texto, útil para EDA e dashboard
    df["resultado"] = df.apply(
        lambda r: "Casa" if r.gols_casa > r.gols_fora
        else ("Fora" if r.gols_casa < r.gols_fora else "Empate"),
        axis=1,
    )
    return df.sort_values(["rodada", "data"]).reset_index(drop=True)


def limpar_jogos_futuros(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza jogos ainda não disputados (sem placar)."""
    df = df.copy()
    if df.empty:
        return df
    df["time_casa"] = df["time_casa"].map(normalizar_nome)
    df["time_fora"] = df["time_fora"].map(normalizar_nome)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["rodada"] = pd.to_numeric(df["rodada"], errors="coerce")
    df = df.dropna(subset=["time_casa", "time_fora"])
    return df.sort_values(["rodada", "data"]).reset_index(drop=True)


def main() -> None:
    garantir_pastas()
    brutas = pd.read_csv(PARTIDAS_RAW)
    partidas = limpar_partidas(brutas)
    tabela = derivar_tabela(partidas)

    partidas.to_csv(PARTIDAS_PROC, index=False, encoding="utf-8")
    tabela.to_csv(TABELA_PROC, index=False, encoding="utf-8")
    print(f"[limpar] {len(partidas)} partidas -> {PARTIDAS_PROC}")
    print(f"[limpar] {len(tabela)} times    -> {TABELA_PROC}")

    # jogos futuros (podem não existir no modo simulação)
    if JOGOS_FUTUROS_RAW.exists():
        futuros_brutos = pd.read_csv(JOGOS_FUTUROS_RAW)
        futuros = limpar_jogos_futuros(futuros_brutos)
        futuros.to_csv(JOGOS_FUTUROS_PROC, index=False, encoding="utf-8")
        print(f"[limpar] {len(futuros)} jogos futuros -> {JOGOS_FUTUROS_PROC}")


if __name__ == "__main__":
    main()
