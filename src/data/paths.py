"""Caminhos centrais do projeto.

Centraliza onde ficam os dados para que todos os scripts apontem para os
mesmos lugares, independente de onde forem executados.
"""
from pathlib import Path

# Raiz do projeto = duas pastas acima deste arquivo (src/data/paths.py)
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"

# Arquivos usados ao longo da Fase 1
PARTIDAS_RAW = RAW / "partidas.csv"
TABELA_RAW = RAW / "tabela_atual.csv"
PARTIDAS_PROC = PROCESSED / "partidas.csv"
TABELA_PROC = PROCESSED / "tabela.csv"
FORMA_PROC = PROCESSED / "forma_recente.csv"


def garantir_pastas() -> None:
    """Cria as pastas de dados/modelos se ainda não existirem."""
    for pasta in (RAW, PROCESSED, MODELS):
        pasta.mkdir(parents=True, exist_ok=True)
