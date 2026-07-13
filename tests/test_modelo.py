"""Testes da Fase 3 — modelo. Rodar: python tests/test_modelo.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.coletar_tabela import simular_temporada
from src.data.limpar import limpar_partidas
from src.features.build_features import FEATURES, construir_dataset, montar_features_atual
from src.models.treinar import split_temporal, treinar

# dados simulados determinísticos (não dependem de internet)
PARTIDAS = limpar_partidas(simular_temporada())
DATASET = construir_dataset(PARTIDAS)
PACOTE = treinar(DATASET)
TIMES = sorted(PARTIDAS["time_casa"].unique())


def test_split_temporal_nao_vaza():
    treino, teste, corte = split_temporal(DATASET)
    assert len(treino) > 0 and len(teste) > 0
    # nenhuma rodada de teste pode ser anterior ao corte (futuro nunca no treino)
    assert treino["rodada"].max() < corte
    assert teste["rodada"].min() >= corte


def test_features_treino_e_previsao_batem():
    # as features montadas "ao vivo" têm exatamente as mesmas colunas do dataset
    feats = montar_features_atual(PARTIDAS, TIMES[0], TIMES[1])
    assert feats is not None
    assert set(feats.keys()) == set(FEATURES)


def test_modelo_supera_acaso():
    # invariante honesto: o modelo escolhido deve superar o chute aleatório de
    # 3 classes (~1/3). "Bater o baseline do mandante" é uma ASPIRAÇÃO que
    # depende dos dados — não um invariante garantido.
    melhor = PACOTE["metricas"][PACOTE["nome_modelo"]]["acuracia"]
    assert melhor > 1 / 3, f"acurácia {melhor:.3f} não supera o acaso"


def test_previsao_probabilidades_validas():
    from src.models.predizer import prever_partida
    ml = prever_partida(TIMES[0], TIMES[1], partidas=PARTIDAS, pacote=PACOTE)
    assert ml is not None
    soma = ml["prob_casa"] + ml["prob_empate"] + ml["prob_fora"]
    assert abs(soma - 1.0) < 1e-6
    assert ml["palpite"] in ("Casa", "Empate", "Fora")


def test_matriz_confusao_formato():
    m = PACOTE["matriz_confusao"]
    assert m.shape == (3, 3)
    # total de jogos na matriz == tamanho do teste
    assert int(m.sum()) == PACOTE["n_teste"]


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_") and callable(fn):
            fn()
    print("OK — todos os testes do modelo passaram.")
