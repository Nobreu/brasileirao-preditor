"""Testes da Fase 2+ — simulador. Rodar: python tests/test_simulador.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.coletar_tabela import simular_temporada
from src.data.limpar import limpar_partidas
from src.features.simulador import (
    forcas_times,
    gols_esperados,
    matriz_placares,
    monte_carlo_partida,
    prever_jogos_futuros,
    probabilidades,
    projetar_temporada,
    simular_partida,
    tabela_projetada,
)

# dados simulados (determinísticos) — não dependem de internet
PARTIDAS = limpar_partidas(simular_temporada())
FORCAS = forcas_times(PARTIDAS)
TIMES = FORCAS.times


def test_forcas_cobre_todos_os_times_e_positivos():
    assert len(TIMES) == 20
    r = FORCAS.ratings
    for col in ("ataque_casa", "defesa_casa", "ataque_fora", "defesa_fora"):
        assert (r[col] > 0).all(), f"rating não-positivo em {col}"


def test_matriz_e_probabilidades_somam_um():
    m = matriz_placares(1.6, 1.1)
    assert abs(m.sum() - 1.0) < 1e-9
    p = probabilidades(m)
    soma = p["prob_casa"] + p["prob_empate"] + p["prob_fora"]
    assert abs(soma - 1.0) < 1e-9


def test_time_forte_favorito_sobre_fraco():
    # ordena por ataque_casa como proxy de força
    ordenado = FORCAS.ratings["ataque_casa"].sort_values()
    fraco, forte = ordenado.index[0], ordenado.index[-1]
    res = simular_partida(PARTIDAS, forte, fraco, forcas=FORCAS)
    assert res["prob_casa"] > res["prob_fora"], "time forte em casa deveria ser favorito"


def test_monte_carlo_bate_analitico():
    casa, fora = TIMES[0], TIMES[1]
    res = simular_partida(PARTIDAS, casa, fora, forcas=FORCAS)
    mc = monte_carlo_partida(res["lam_casa"], res["lam_fora"], n=40000, seed=1)
    # diferença analítico vs Monte Carlo pequena
    assert abs(mc["freq_casa"] - res["prob_casa"]) < 0.02
    assert abs(mc["freq_empate"] - res["prob_empate"]) < 0.02


def test_gols_esperados_positivos():
    lc, lf = gols_esperados(FORCAS, TIMES[0], TIMES[1])
    assert lc > 0 and lf > 0


def test_projecao_invariantes():
    # cria jogos futuros fictícios: um returno extra entre alguns times
    futuros = pd.DataFrame(
        [
            {"rodada": 99, "data": "2024-12-01", "time_casa": TIMES[i], "time_fora": TIMES[-1 - i]}
            for i in range(10)
        ]
    )
    proj = projetar_temporada(PARTIDAS, futuros, n_sims=2000, seed=7)
    assert len(proj) == 20
    # probabilidades em [0, 1]
    for c in ("prob_titulo", "prob_g4", "prob_rebaixamento"):
        assert proj[c].between(0, 1).all()
    # invariante forte: exatamente 1 campeão por simulação
    assert abs(proj["prob_titulo"].sum() - 1.0) < 1e-9
    # exatamente 4 rebaixados por simulação
    assert abs(proj["prob_rebaixamento"].sum() - 4.0) < 1e-9


def test_projecao_vazia_sem_futuros():
    vazio = pd.DataFrame(columns=["rodada", "data", "time_casa", "time_fora"])
    assert projetar_temporada(PARTIDAS, vazio, n_sims=100).empty


def _futuros_ficticios():
    return pd.DataFrame(
        [
            {"rodada": 99, "data": "2024-12-01", "time_casa": TIMES[i], "time_fora": TIMES[-1 - i]}
            for i in range(10)
        ]
    )


def test_prever_jogos_futuros():
    palp = prever_jogos_futuros(PARTIDAS, _futuros_ficticios(), forcas=FORCAS)
    assert len(palp) == 10
    # probabilidades de cada jogo somam ~1
    somas = palp["prob_casa"] + palp["prob_empate"] + palp["prob_fora"]
    assert (somas.sub(1.0).abs() < 1e-9).all()
    assert palp["tendencia"].isin(["Casa", "Empate", "Fora"]).all()


def test_tabela_projetada_soma_pontos():
    futuros = _futuros_ficticios()
    tab, palp = tabela_projetada(PARTIDAS, futuros, forcas=FORCAS)
    assert len(tab) == 20
    # projetar só ADICIONA jogos -> pontos finais >= pontos atuais
    atual = derivar_tabela_local()
    for t in TIMES:
        p_final = tab.set_index("time").at[t, "P"]
        p_atual = atual.set_index("time").at[t, "P"]
        assert p_final >= p_atual, f"{t}: projetado {p_final} < atual {p_atual}"


def derivar_tabela_local():
    from src.data.coletar_tabela import derivar_tabela
    return derivar_tabela(PARTIDAS)


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
    print("OK — todos os testes do simulador passaram.")
