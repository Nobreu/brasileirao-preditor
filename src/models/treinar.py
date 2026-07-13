"""Fase 3 — Treino do modelo preditivo.

Treina um classificador multiclasse (Casa / Empate / Fora) a partir do
`dataset_modelagem.csv` da Fase 2. Pontos-chave da metodologia:

- **Split temporal:** treino nas rodadas antigas, teste nas recentes (NUNCA
  embaralhar — prever futebol é prever o futuro).
- **Baseline ingênuo:** "o mandante sempre vence" (régua a ser superada).
- **Dois modelos:** Regressão Logística (simples) e Random Forest (robusto).
- **Métricas:** acurácia, log loss e matriz de confusão.
- Persiste o melhor modelo em `models/modelo_v1.pkl` (com joblib).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..data.paths import DATASET_MODELAGEM, MODELO_V1, garantir_pastas
from ..features.build_features import FEATURES

CLASSES = ["Casa", "Empate", "Fora"]
FRACAO_TESTE = 0.30   # ~30% das rodadas mais recentes viram teste


def split_temporal(dataset: pd.DataFrame, fracao_teste: float = FRACAO_TESTE):
    """Separa treino/teste por rodada (as mais recentes ficam no teste)."""
    ds = dataset.sort_values(["rodada"]).reset_index(drop=True)
    rodadas = sorted(ds["rodada"].unique())
    corte = rodadas[int(len(rodadas) * (1 - fracao_teste))]
    treino = ds[ds["rodada"] < corte]
    teste = ds[ds["rodada"] >= corte]
    return treino, teste, corte


def _avaliar(modelo, X_te, y_te) -> dict:
    pred = modelo.predict(X_te)
    proba = modelo.predict_proba(X_te)
    return {
        "acuracia": float(accuracy_score(y_te, pred)),
        "log_loss": float(log_loss(y_te, proba, labels=modelo.classes_)),
    }


def treinar(dataset: pd.DataFrame | None = None) -> dict:
    """Treina os modelos e devolve um 'pacote' com tudo que o app precisa."""
    if dataset is None:
        dataset = pd.read_csv(DATASET_MODELAGEM)

    treino, teste, corte = split_temporal(dataset)
    X_tr, y_tr = treino[FEATURES], treino["alvo"]
    X_te, y_te = teste[FEATURES], teste["alvo"]

    # ---- Baseline ingênuo: prever sempre "Casa" ----
    baseline_acc = float((y_te == "Casa").mean())

    # ---- Regressão Logística (padroniza as features) ----
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ).fit(X_tr, y_tr)

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced_subsample"
    ).fit(X_tr, y_tr)

    metricas = {
        "Regressão Logística": _avaliar(logreg, X_te, y_te),
        "Random Forest": _avaliar(rf, X_te, y_te),
    }

    # escolhe o melhor por acurácia (desempate: menor log loss)
    nome_melhor = max(
        metricas, key=lambda m: (metricas[m]["acuracia"], -metricas[m]["log_loss"])
    )
    melhor = logreg if nome_melhor == "Regressão Logística" else rf

    # importância das features (sempre do Random Forest, que a expõe direto)
    importancias = (
        pd.Series(rf.feature_importances_, index=FEATURES)
        .sort_values(ascending=False)
    )

    matriz = confusion_matrix(y_te, melhor.predict(X_te), labels=CLASSES)

    pacote = {
        "modelo": melhor,
        "nome_modelo": nome_melhor,
        "features": FEATURES,
        "classes": list(melhor.classes_),
        "metricas": metricas,
        "baseline_acuracia": baseline_acc,
        "importancias": importancias,
        "matriz_confusao": matriz,
        "rodada_corte": int(corte),
        "n_treino": int(len(treino)),
        "n_teste": int(len(teste)),
    }
    return pacote


def treinar_e_salvar() -> dict:
    garantir_pastas()
    pacote = treinar()
    joblib.dump(pacote, MODELO_V1)
    return pacote


def main() -> None:
    pacote = treinar_e_salvar()
    print(f"[treinar] modelo salvo -> {MODELO_V1}")
    print(f"[treinar] split: treino={pacote['n_treino']} teste={pacote['n_teste']} "
          f"(corte na rodada {pacote['rodada_corte']})")
    print(f"[treinar] baseline (mandante sempre vence): {pacote['baseline_acuracia']:.1%}")
    for nome, m in pacote["metricas"].items():
        marca = "  <== escolhido" if nome == pacote["nome_modelo"] else ""
        print(f"[treinar] {nome:22s} acurácia={m['acuracia']:.1%}  log_loss={m['log_loss']:.3f}{marca}")
    print("[treinar] top 5 features:")
    for feat, imp in pacote["importancias"].head(5).items():
        print(f"           {feat:24s} {imp:.3f}")


if __name__ == "__main__":
    main()
