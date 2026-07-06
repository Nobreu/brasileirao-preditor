"""Pipeline completo: coleta -> limpeza -> forma recente -> features.

Uso:
    python -m src.data.preparar_dados

Também é chamado automaticamente pelo dashboard quando os dados
processados ainda não existem.

Evolução por fase (bola de neve):
- Fase 1: coleta -> limpeza -> forma recente
- Fase 2: + engenharia de features (dataset de modelagem)
"""
from __future__ import annotations

from . import coletar_tabela, forma, limpar
from .paths import DATASET_MODELAGEM, FORMA_PROC, PARTIDAS_PROC, TABELA_PROC


def executar(forcar: bool = False) -> None:
    """Roda o pipeline inteiro.

    Se `forcar` for False e os arquivos processados já existirem, não faz nada.
    """
    processados = (PARTIDAS_PROC, TABELA_PROC, FORMA_PROC, DATASET_MODELAGEM)
    if not forcar and all(p.exists() for p in processados):
        print("[preparar_dados] Dados já existem. Use forcar=True para refazer.")
        return
    coletar_tabela.main()
    limpar.main()
    forma.main()
    # Fase 2: importado aqui dentro para não criar dependência no topo do módulo
    from ..features import build_features
    build_features.main()
    print("[preparar_dados] Pipeline concluído.")


if __name__ == "__main__":
    executar(forcar=True)
