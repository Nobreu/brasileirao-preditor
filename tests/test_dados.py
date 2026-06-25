"""Testes simples da Fase 1 — rodar com: python -m pytest -q (ou python tests/test_dados.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.coletar_tabela import derivar_tabela, simular_temporada
from src.data.forma import calcular_forma
from src.data.limpar import limpar_partidas
from src.data.times import TIMES_SERIE_A, normalizar_nome


def test_simulacao_tem_todos_os_jogos():
    p = simular_temporada()
    # 20 times, turno e returno = 38 rodadas * 10 jogos = 380
    assert len(p) == 380
    assert set(p["time_casa"]) == set(TIMES_SERIE_A)


def test_tabela_tem_20_times_e_pontos_validos():
    tabela = derivar_tabela(simular_temporada())
    assert len(tabela) == 20
    # cada time jogou 38 partidas
    assert (tabela["J"] == 38).all()
    # pontos = 3V + E
    assert (tabela["P"] == 3 * tabela["V"] + tabela["E"]).all()


def test_normalizar_nome():
    assert normalizar_nome("Sao Paulo") == "São Paulo"
    assert normalizar_nome("RED BULL BRAGANTINO") == "Bragantino"
    assert normalizar_nome("inter") == "Internacional"


def test_forma_no_maximo_5_jogos():
    partidas = limpar_partidas(simular_temporada())
    forma = calcular_forma(partidas, n=5)
    assert len(forma) == 20
    assert (forma["n_jogos"] <= 5).all()
    # pontos coerentes com a sequência
    assert (forma["pontos_ultimos_n"] == 3 * forma["vitorias"] + forma["empates"]).all()


if __name__ == "__main__":
    test_simulacao_tem_todos_os_jogos()
    test_tabela_tem_20_times_e_pontos_validos()
    test_normalizar_nome()
    test_forma_no_maximo_5_jogos()
    print("OK — todos os testes passaram.")
