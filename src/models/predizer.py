"""Fase 3 — Previsão de partidas com o modelo treinado.

Carrega o modelo salvo (treina na hora se ainda não existir) e prevê o
resultado de um confronto qualquer a partir do estado ATUAL dos dois times.
"""
from __future__ import annotations

import joblib
import pandas as pd

from ..data.paths import MODELO_V1, PARTIDAS_PROC
from ..features.build_features import montar_features_atual


def carregar_modelo() -> dict:
    """Devolve o pacote do modelo; treina e salva caso não exista ainda."""
    if not MODELO_V1.exists():
        from . import treinar
        return treinar.treinar_e_salvar()
    return joblib.load(MODELO_V1)


def prever_partida(casa: str, fora: str,
                   partidas: pd.DataFrame | None = None,
                   pacote: dict | None = None) -> dict | None:
    """Prevê Casa/Empate/Fora para um confronto usando o estado atual dos times.

    Devolve dict com as probabilidades por classe + o palpite. None se algum
    time não tiver histórico.
    """
    if partidas is None:
        partidas = pd.read_csv(PARTIDAS_PROC)
    pacote = pacote or carregar_modelo()

    feats = montar_features_atual(partidas, casa, fora)
    if feats is None:
        return None

    X = pd.DataFrame([feats])[pacote["features"]]
    proba = pacote["modelo"].predict_proba(X)[0]
    probs = {classe: float(p) for classe, p in zip(pacote["modelo"].classes_, proba)}
    palpite = max(probs, key=probs.get)
    return {
        "time_casa": casa,
        "time_fora": fora,
        "prob_casa": probs.get("Casa", 0.0),
        "prob_empate": probs.get("Empate", 0.0),
        "prob_fora": probs.get("Fora", 0.0),
        "palpite": palpite,
        "modelo": pacote["nome_modelo"],
    }
