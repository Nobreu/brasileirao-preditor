"""Pipeline completo da Fase 1: coleta -> limpeza -> forma recente.

Uso:
    python -m src.data.preparar_dados

Também é chamado automaticamente pelo dashboard quando os dados
processados ainda não existem.
"""
from __future__ import annotations

from . import coletar_tabela, forma, limpar
from .paths import FORMA_PROC, PARTIDAS_PROC, TABELA_PROC


def executar(forcar: bool = False) -> None:
    """Roda o pipeline inteiro.

    Se `forcar` for False e os arquivos processados já existirem, não faz nada.
    """
    processados = (PARTIDAS_PROC, TABELA_PROC, FORMA_PROC)
    if not forcar and all(p.exists() for p in processados):
        print("[preparar_dados] Dados já existem. Use forcar=True para refazer.")
        return
    coletar_tabela.main()
    limpar.main()
    forma.main()
    print("[preparar_dados] Pipeline concluído.")


if __name__ == "__main__":
    executar(forcar=True)
