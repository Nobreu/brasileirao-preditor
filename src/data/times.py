"""Times da Série A e utilidades de padronização de nomes.

Mantemos aqui a lista canônica dos 20 clubes e um mapa de apelidos para
normalizar nomes que vêm de fontes diferentes (API, raspagem, planilhas).
"""
from __future__ import annotations

# Lista canônica (Série A 2024) — é a "fonte da verdade" dos nomes.
TIMES_SERIE_A = [
    "Athletico-PR",
    "Atlético-GO",
    "Atlético-MG",
    "Bahia",
    "Botafogo",
    "Bragantino",
    "Corinthians",
    "Criciúma",
    "Cruzeiro",
    "Cuiabá",
    "Flamengo",
    "Fluminense",
    "Fortaleza",
    "Grêmio",
    "Internacional",
    "Juventude",
    "Palmeiras",
    "São Paulo",
    "Vasco",
    "Vitória",
]

# Força relativa aproximada (apenas para a simulação de dados da Fase 1).
# Não é estatística real — serve para os resultados simulados ficarem
# plausíveis. Valores maiores = time tende a marcar mais e sofrer menos.
FORCA = {
    "Athletico-PR": 1.15,
    "Atlético-GO": 0.80,
    "Atlético-MG": 1.25,
    "Bahia": 1.10,
    "Botafogo": 1.40,
    "Bragantino": 1.05,
    "Corinthians": 1.05,
    "Criciúma": 0.85,
    "Cruzeiro": 1.10,
    "Cuiabá": 0.80,
    "Flamengo": 1.35,
    "Fluminense": 1.00,
    "Fortaleza": 1.20,
    "Grêmio": 1.00,
    "Internacional": 1.20,
    "Juventude": 0.85,
    "Palmeiras": 1.35,
    "São Paulo": 1.20,
    "Vasco": 0.95,
    "Vitória": 0.85,
}

# Apelidos comuns -> nome canônico. Adicione conforme novas fontes aparecem.
APELIDOS = {
    "athletico paranaense": "Athletico-PR",
    "athletico-pr": "Athletico-PR",
    "athletico pr": "Athletico-PR",
    "cap": "Athletico-PR",
    "atletico goianiense": "Atlético-GO",
    "atletico-go": "Atlético-GO",
    "atletico go": "Atlético-GO",
    "atletico mineiro": "Atlético-MG",
    "atletico-mg": "Atlético-MG",
    "atletico mg": "Atlético-MG",
    "galo": "Atlético-MG",
    "ec bahia": "Bahia",
    "botafogo fr": "Botafogo",
    "botafogo rj": "Botafogo",
    "red bull bragantino": "Bragantino",
    "rb bragantino": "Bragantino",
    "bragantino": "Bragantino",
    "sc corinthians": "Corinthians",
    "corinthians paulista": "Corinthians",
    "criciuma": "Criciúma",
    "cruzeiro ec": "Cruzeiro",
    "cuiaba": "Cuiabá",
    "cuiaba ec": "Cuiabá",
    "flamengo rj": "Flamengo",
    "cr flamengo": "Flamengo",
    "fluminense fc": "Fluminense",
    "fortaleza ec": "Fortaleza",
    "gremio": "Grêmio",
    "internacional rs": "Internacional",
    "inter": "Internacional",
    "sc internacional": "Internacional",
    "juventude rs": "Juventude",
    "ec juventude": "Juventude",
    "se palmeiras": "Palmeiras",
    "sao paulo": "São Paulo",
    "sao paulo fc": "São Paulo",
    "vasco da gama": "Vasco",
    "cr vasco da gama": "Vasco",
    "ec vitoria": "Vitória",
    "vitoria": "Vitória",
    # --- Nomes oficiais como vêm da API football-data.org (v4) ---
    "botafogo fr": "Botafogo",
    "ca mineiro": "Atlético-MG",
    "ca paranaense": "Athletico-PR",
    "cr flamengo": "Flamengo",
    "cr vasco da gama": "Vasco",
    "chapecoense af": "Chapecoense",
    "clube do remo": "Remo",
    "coritiba fbc": "Coritiba",
    "cruzeiro ec": "Cruzeiro",
    "ec bahia": "Bahia",
    "ec vitoria ": "Vitória",
    "fluminense fc": "Fluminense",
    "gremio fbpa": "Grêmio",
    "mirassol fc": "Mirassol",
    "rb bragantino": "Bragantino",
    "sc corinthians paulista": "Corinthians",
    "sc internacional": "Internacional",
    "se palmeiras": "Palmeiras",
    "santos fc": "Santos",
    "sao paulo fc": "São Paulo",
}


def normalizar_nome(nome: str) -> str:
    """Devolve o nome canônico de um time.

    Tenta casar pelo apelido (sem acento/caixa). Se não achar, devolve o
    nome original com espaços limpos — assim nada é perdido silenciosamente.
    """
    if not isinstance(nome, str):
        return nome
    chave = nome.strip().lower()
    # remoção simples de acentos para casar apelidos
    import unicodedata

    chave_sem_acento = (
        unicodedata.normalize("NFKD", chave)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    if chave_sem_acento in APELIDOS:
        return APELIDOS[chave_sem_acento]
    # já é canônico?
    for canon in TIMES_SERIE_A:
        if chave_sem_acento == (
            unicodedata.normalize("NFKD", canon.lower())
            .encode("ascii", "ignore")
            .decode("ascii")
        ):
            return canon
    return nome.strip()
