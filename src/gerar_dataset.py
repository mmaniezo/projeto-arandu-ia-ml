"""
Geração do dataset SINTÉTICO do Arandu.

As fontes reais de dados do Arandu seriam NASA POWER, INMET, IBGE/PAM e NDVI
Sentinel-2. Aqui usa-se um dataset sintético agronomicamente fundamentado, com
as mesmas 7 features e o mesmo alvo de 3 classes, de modo que o modelo aceite
dados reais no futuro sem alteração de contrato.

Lógica do rótulo: um índice latente de estresse hídrico é calculado a partir de
relações agronômicas conhecidas (balanço hídrico, vigor vegetativo, fenologia e
sensibilidade da cultura). O índice recebe ruído gaussiano e é discretizado em
3 classes por quantis, produzindo desbalanceamento realista (mais "Baixo" que
"Alto", como observado em campo) e fronteiras não-triviais entre classes.
"""

import os
import numpy as np
import pandas as pd

from config import SEED, FEATURES, ALVO, CULTURAS, FAIXAS, ARQ_DATASET

N_AMOSTRAS = 3000


def _norm(x, lo, hi):
    """Normaliza para [0, 1] e satura fora da faixa."""
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _indice_estresse(df, rng):
    """Índice latente de estresse hídrico em [0, 1] (maior = pior).

    Combina termos agronômicos; pesos somam ~1 antes dos modificadores.
    """
    # Vigor vegetativo: NDVI baixo e/ou em queda => mais estresse.
    estresse_ndvi = 1.0 - _norm(df["ndvi_atual"], 0.10, 0.92)
    queda_ndvi = _norm(-df["variacao_ndvi_15d"], 0.0, 0.30)  # só quedas pesam

    # Demanda atmosférica: calor e radiação elevam a evapotranspiração.
    termo_temp = _norm(df["temperatura_media_7d"], 15.0, 40.0)
    termo_rad = _norm(df["radiacao_solar_media"], 10.0, 30.0)

    # Oferta hídrica: chuva acumulada baixa => déficit (satura em 200 mm).
    deficit_chuva = 1.0 - _norm(df["precipitacao_acumulada_30d"], 0.0, 200.0)

    base = (
        0.28 * estresse_ndvi
        + 0.14 * queda_ndvi
        + 0.18 * termo_temp
        + 0.10 * termo_rad
        + 0.30 * deficit_chuva
    )

    # Modificador fenológico: pico de sensibilidade na fase reprodutiva
    # (~70 dias após o plantio), modelado como sino gaussiano.
    dias = df["dias_apos_plantio"].to_numpy(dtype=float)
    fenologia = 0.85 + 0.30 * np.exp(-((dias - 70.0) ** 2) / (2 * 30.0 ** 2))

    # Sensibilidade hídrica da cultura (tabela CULTURAS).
    sens = df["tipo_cultura"].map({k: v[1] for k, v in CULTURAS.items()}).to_numpy()
    mod_cultura = 0.80 + 0.40 * sens  # ~[0.80, 1.18]

    indice = base.to_numpy() * fenologia * mod_cultura

    # Ruído gaussiano: torna as fronteiras entre classes não-determinísticas.
    indice = indice + rng.normal(0.0, 0.06, size=len(df))
    return _norm(indice, indice.min(), indice.max())


def gerar(n=N_AMOSTRAS, seed=SEED):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "ndvi_atual": rng.uniform(*FAIXAS["ndvi_atual"], n),
        "variacao_ndvi_15d": rng.uniform(*FAIXAS["variacao_ndvi_15d"], n),
        "temperatura_media_7d": rng.uniform(*FAIXAS["temperatura_media_7d"], n),
        "precipitacao_acumulada_30d": rng.uniform(*FAIXAS["precipitacao_acumulada_30d"], n),
        "radiacao_solar_media": rng.uniform(*FAIXAS["radiacao_solar_media"], n),
        "tipo_cultura": rng.integers(min(CULTURAS), max(CULTURAS) + 1, n),
        "dias_apos_plantio": rng.integers(*FAIXAS["dias_apos_plantio"], n),
    })

    indice = _indice_estresse(df, rng)

    # Discretização por quantis -> desbalanceamento realista:
    # ~60% Baixo, ~25% Moderado, ~15% Alto.
    q_baixo, q_alto = np.quantile(indice, [0.60, 0.85])
    risco = np.where(indice <= q_baixo, 0, np.where(indice <= q_alto, 1, 2))
    df[ALVO] = risco.astype(int)

    # Reordena colunas: features na ordem canônica + alvo.
    df = df[FEATURES + [ALVO]]
    return df


def main():
    df = gerar()
    os.makedirs(os.path.dirname(ARQ_DATASET), exist_ok=True)
    df.to_csv(ARQ_DATASET, index=False)

    dist = df[ALVO].value_counts().sort_index()
    print(f"Dataset gerado: {len(df)} amostras -> {ARQ_DATASET}")
    print("Distribuição do alvo (risco_estresse):")
    for cls, qtd in dist.items():
        print(f"  classe {cls}: {qtd} ({qtd / len(df):.1%})")


if __name__ == "__main__":
    main()
