"""Testes da Fase 2 — features. Rodar: python tests/test_features.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.coletar_tabela import simular_temporada
from src.data.limpar import limpar_partidas
from src.features.build_features import construir_dataset, resumo_times

# usa dados simulados (determinísticos) para o teste não depender da internet
PARTIDAS = limpar_partidas(simular_temporada())
COLS_META = {"rodada", "data", "time_casa", "time_fora", "alvo"}


def test_dataset_tem_pelo_menos_8_features():
    ds = construir_dataset(PARTIDAS)
    features = [c for c in ds.columns if c not in COLS_META]
    assert len(features) >= 8, f"esperado >=8 features, veio {len(features)}"


def test_alvo_tem_tres_classes():
    ds = construir_dataset(PARTIDAS)
    assert set(ds["alvo"].unique()) <= {"Casa", "Empate", "Fora"}
    assert ds["alvo"].nunique() == 3


def test_sem_data_leak():
    """Prova de ausência de vazamento.

    As features de uma partida só podem depender de jogos ANTERIORES. Então,
    construir o dataset usando apenas os jogos até a partida k deve produzir,
    para a partida k, EXATAMENTE a mesma linha que o dataset completo produz.
    Se um jogo futuro influenciasse a linha, os valores mudariam.
    """
    p = PARTIDAS.copy()
    p["data"] = pd.to_datetime(p["data"], errors="coerce")
    p = p.sort_values(["rodada", "data"]).reset_index(drop=True)

    completo = construir_dataset(p)
    feat_cols = [c for c in completo.columns if c not in ("data",)]

    verificadas = 0
    # verifica 3 partidas espalhadas no meio/fim do campeonato
    for k in (120, 200, 300):
        prefixo = construir_dataset(p.iloc[: k + 1])   # só até a partida k
        if prefixo.empty:
            continue
        ultima = prefixo.iloc[-1]
        # acha a mesma partida no dataset completo
        match = completo[
            (completo["rodada"] == ultima["rodada"])
            & (completo["time_casa"] == ultima["time_casa"])
            & (completo["time_fora"] == ultima["time_fora"])
        ]
        assert len(match) == 1
        completo_row = match.iloc[0]
        for c in feat_cols:
            assert completo_row[c] == ultima[c], f"data leak na coluna {c} (partida k={k})"
        verificadas += 1
    assert verificadas >= 1, "nenhuma partida pôde ser verificada"


def test_resumo_times_tem_20():
    r = resumo_times(PARTIDAS)
    assert len(r) == 20
    assert (r["aprov_casa"].between(0, 1)).all()
    assert (r["aprov_fora"].between(0, 1)).all()


if __name__ == "__main__":
    test_dataset_tem_pelo_menos_8_features()
    test_alvo_tem_tres_classes()
    test_sem_data_leak()
    test_resumo_times_tem_20()
    print("OK — todos os testes da Fase 2 passaram (inclusive o de data leak).")
