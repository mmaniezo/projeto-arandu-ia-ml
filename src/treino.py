"""
Treinamento e avaliação do modelo de risco do Arandu.

Compara quatro algoritmos de classificação supervisionada:
  - Regressão Logística (modelo linear, com análise de coeficientes)
  - Naive Bayes (GaussianNB)
  - Árvore de Decisão
  - MLP (rede neural rasa)
Inclui ainda KMeans como análise exploratória não supervisionada.

Métricas: acurácia, F1-macro, matriz de confusão e importância das features.
Meta de qualidade: acurácia >= 75% em validação cruzada.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

from config import (
    SEED, FEATURES, FEATURE_CATEGORICA, ALVO, CLASSES,
    ARQ_DATASET, ARQ_MODELO, DIR_FIGURAS, ARQ_METRICAS,
)

NUMERICAS = [f for f in FEATURES if f != FEATURE_CATEGORICA]
NOMES_CLASSES = [CLASSES[i] for i in sorted(CLASSES)]


def preprocessador():
    """One-Hot na cultura + padronização das numéricas (preparação dos dados)."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         [FEATURE_CATEGORICA]),
    ])


def construir_modelos():
    """Um Pipeline (preprocessamento + estimador) por algoritmo avaliado."""
    return {
        "Regressão Logística": LogisticRegression(max_iter=2000, random_state=SEED),
        "Naive Bayes": GaussianNB(),
        "Árvore de Decisão": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, random_state=SEED),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=1000,
            early_stopping=True, random_state=SEED),
    }


def avaliar_cv(X, y):
    """Validação cruzada estratificada (5 folds): acurácia + F1-macro por modelo."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    resultados = {}
    for nome, est in construir_modelos().items():
        pipe = Pipeline([("prep", preprocessador()), ("clf", est)])
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring={"acc": "accuracy", "f1m": "f1_macro"},
        )
        resultados[nome] = {
            "acuracia_cv": float(scores["test_acc"].mean()),
            "acuracia_cv_std": float(scores["test_acc"].std()),
            "f1_macro_cv": float(scores["test_f1m"].mean()),
            "f1_macro_cv_std": float(scores["test_f1m"].std()),
        }
        print(f"  {nome:22s} | acc {resultados[nome]['acuracia_cv']:.3f} "
              f"± {resultados[nome]['acuracia_cv_std']:.3f} | "
              f"F1-macro {resultados[nome]['f1_macro_cv']:.3f}")
    return resultados


def plot_comparacao(resultados, caminho):
    nomes = list(resultados)
    acc = [resultados[n]["acuracia_cv"] for n in nomes]
    f1m = [resultados[n]["f1_macro_cv"] for n in nomes]
    x = np.arange(len(nomes))
    plt.figure(figsize=(9, 5))
    plt.bar(x - 0.2, acc, 0.4, label="Acurácia")
    plt.bar(x + 0.2, f1m, 0.4, label="F1-macro")
    plt.axhline(0.75, color="red", ls="--", lw=1, label="Meta (75%)")
    plt.xticks(x, nomes, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Score (validação cruzada 5-fold)")
    plt.title("Comparação de modelos de risco agrícola")
    plt.legend()
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


def plot_matriz_confusao(y_true, y_pred, titulo, caminho):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES)
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()


def plot_importancia(pipe, X_test, y_test, caminho):
    """Importância por permutação (modelo-agnóstica) sobre as features originais."""
    r = permutation_importance(
        pipe, X_test, y_test, n_repeats=20, random_state=SEED, scoring="f1_macro")
    ordem = np.argsort(r.importances_mean)
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(FEATURES)[ordem], r.importances_mean[ordem],
             xerr=r.importances_std[ordem], color="seagreen")
    plt.xlabel("Queda média de F1-macro ao permutar a feature")
    plt.title("Importância das features (permutação)")
    plt.tight_layout()
    plt.savefig(caminho, dpi=130)
    plt.close()
    return {FEATURES[i]: float(r.importances_mean[i]) for i in range(len(FEATURES))}


def analise_coeficientes(X_train, y_train):
    """Análise de coeficientes da Regressão Logística (modelo linear)."""
    pipe = Pipeline([("prep", preprocessador()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])
    pipe.fit(X_train, y_train)
    nomes_feat = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_  # (n_classes, n_features)
    tabela = {}
    for idx, cls in enumerate(sorted(CLASSES)):
        tabela[CLASSES[cls]] = {
            nomes_feat[j]: float(coefs[idx, j]) for j in range(len(nomes_feat))
        }
    return tabela


def clustering_exploratorio(X_train):
    """KMeans (k=3) como análise não supervisionada exploratória das features."""
    prep = preprocessador().fit(X_train)
    Xt = prep.transform(X_train)
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED).fit(Xt)
    _, contagem = np.unique(km.labels_, return_counts=True)
    return {"inercia": float(km.inertia_),
            "tamanho_clusters": contagem.tolist()}


def main():
    os.makedirs(DIR_FIGURAS, exist_ok=True)
    os.makedirs(os.path.dirname(ARQ_MODELO), exist_ok=True)

    df = pd.read_csv(ARQ_DATASET)
    X, y = df[FEATURES], df[ALVO]

    # Hold-out estratificado para avaliação final (20%).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    print("Validação cruzada (5-fold):")
    resultados = avaliar_cv(X_train, y_train)

    # Seleção do melhor modelo por F1-macro (essencial pelo desbalanceamento de classes).
    melhor_nome = max(resultados, key=lambda n: resultados[n]["f1_macro_cv"])
    print(f"\nMelhor modelo por F1-macro: {melhor_nome}")

    melhor = Pipeline([("prep", preprocessador()),
                       ("clf", construir_modelos()[melhor_nome])])
    melhor.fit(X_train, y_train)
    y_pred = melhor.predict(X_test)

    acc_test = accuracy_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred, average="macro")
    print(f"Hold-out  -> acurácia {acc_test:.3f} | F1-macro {f1_test:.3f}")
    print("\nRelatório de classificação (hold-out):")
    print(classification_report(y_test, y_pred, target_names=NOMES_CLASSES))

    # Figuras de avaliação visual.
    plot_comparacao(resultados, f"{DIR_FIGURAS}/comparacao_modelos.png")
    plot_matriz_confusao(
        y_test, y_pred, f"Matriz de Confusão — {melhor_nome}",
        f"{DIR_FIGURAS}/matriz_confusao.png")
    importancias = plot_importancia(
        melhor, X_test, y_test, f"{DIR_FIGURAS}/importancia_features.png")

    coefs = analise_coeficientes(X_train, y_train)
    cluster = clustering_exploratorio(X_train)

    # Persistência do modelo com Joblib.
    joblib.dump(melhor, ARQ_MODELO)
    print(f"\nModelo salvo em {ARQ_MODELO}")

    metricas = {
        "melhor_modelo": melhor_nome,
        "validacao_cruzada": resultados,
        "holdout": {"acuracia": float(acc_test), "f1_macro": float(f1_test)},
        "atende_meta_75pct": bool(resultados[melhor_nome]["acuracia_cv"] >= 0.75),
        "importancia_features": importancias,
        "coeficientes_regressao_logistica": coefs,
        "clustering_kmeans": cluster,
    }
    with open(ARQ_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"Métricas salvas em {ARQ_METRICAS}")


if __name__ == "__main__":
    main()
