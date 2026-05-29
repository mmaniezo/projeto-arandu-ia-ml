"""
Demonstração de inferência do modelo de risco do Arandu.

Carrega o modelo serializado e classifica novos diagnósticos a partir das 7
features de entrada. Representa, de forma mínima, o que o microsserviço Python
entregaria ao backend, sem expor um servidor HTTP.
"""

import pandas as pd
import joblib

from config import FEATURES, CLASSES, ARQ_MODELO

# Exemplos: uma lavoura saudável e uma sob déficit hídrico severo.
EXEMPLOS = [
    {  # esperado: risco Baixo
        "ndvi_atual": 0.80, "variacao_ndvi_15d": 0.05,
        "temperatura_media_7d": 24.0, "precipitacao_acumulada_30d": 160.0,
        "radiacao_solar_media": 17.0, "tipo_cultura": 1, "dias_apos_plantio": 40,
    },
    {  # esperado: risco Alto
        "ndvi_atual": 0.28, "variacao_ndvi_15d": -0.18,
        "temperatura_media_7d": 36.0, "precipitacao_acumulada_30d": 12.0,
        "radiacao_solar_media": 27.0, "tipo_cultura": 2, "dias_apos_plantio": 70,
    },
]


def main():
    modelo = joblib.load(ARQ_MODELO)
    X = pd.DataFrame(EXEMPLOS)[FEATURES]
    preds = modelo.predict(X)
    probs = modelo.predict_proba(X)
    for i, p in enumerate(preds):
        conf = probs[i][p]
        print(f"Exemplo {i + 1}: risco = {CLASSES[int(p)]} "
              f"(classe {int(p)}, confiança {conf:.1%})")


if __name__ == "__main__":
    main()
