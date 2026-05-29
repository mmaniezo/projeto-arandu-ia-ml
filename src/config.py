"""
Configuração central do componente de Machine Learning do Arandu.

Define, em um único lugar, as 7 features de entrada, a variável-alvo e a
codificação do tipo de cultura usadas em todo o pipeline.
"""

# Reprodutibilidade: semente fixa em todo o pipeline.
SEED = 42

# --- Features de entrada ---
FEATURES = [
    "ndvi_atual",                 # NDVI da última leitura
    "variacao_ndvi_15d",          # Tendência do NDVI nos últimos 15 dias
    "temperatura_media_7d",       # Média de temperatura na semana (°C)
    "precipitacao_acumulada_30d", # Chuva acumulada em 30 dias (mm)
    "radiacao_solar_media",       # Insolação média do período (MJ/m²/dia)
    "tipo_cultura",               # Categórica codificada (ver CULTURAS)
    "dias_apos_plantio",          # Idade da lavoura em dias
]

# Feature categórica que exige One-Hot na preparação dos dados.
FEATURE_CATEGORICA = "tipo_cultura"

# --- Variável-alvo ---
ALVO = "risco_estresse"
CLASSES = {0: "Baixo", 1: "Moderado", 2: "Alto"}

# --- Codificação do tipo de cultura ---
# Inteiro -> (nome, sensibilidade hídrica relativa).
# A sensibilidade (0=tolerante, 1=sensível à seca) é usada apenas na geração
# sintética do rótulo; o modelo recebe somente o código categórico.
CULTURAS = {
    1: ("milho",  0.85),
    2: ("feijao", 0.95),
    3: ("cafe",   0.55),
    4: ("cana",   0.45),
    5: ("soja",   0.75),
}

# Faixas físicas plausíveis de cada feature contínua (mín, máx).
# Usadas para gerar dados sintéticos dentro de limites agronômicos realistas.
FAIXAS = {
    "ndvi_atual":                 (0.10, 0.92),
    "variacao_ndvi_15d":          (-0.30, 0.30),
    "temperatura_media_7d":       (15.0, 40.0),
    "precipitacao_acumulada_30d": (0.0, 300.0),
    "radiacao_solar_media":       (10.0, 30.0),
    "dias_apos_plantio":          (0, 150),
}

# Caminhos de saída do pipeline, ancorados na raiz do projeto
# (independem do diretório de execução).
import os as _os
_RAIZ = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

ARQ_DATASET = _os.path.join(_RAIZ, "data", "dataset_arandu.csv")
ARQ_MODELO = _os.path.join(_RAIZ, "modelos", "modelo_risco.joblib")
DIR_FIGURAS = _os.path.join(_RAIZ, "relatorio", "figuras")
ARQ_METRICAS = _os.path.join(_RAIZ, "relatorio", "metricas.json")
