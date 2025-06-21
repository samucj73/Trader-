from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

from captura_api import router as captura_router
app.include_router(captura_router)

HISTORICO_PATH = "historico_coluna_duzia.json"

def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def get_duzia(n):
    if n == 0:
        return 0
    elif 1 <= n <= 12:
        return 1
    elif 13 <= n <= 24:
        return 2
    elif 25 <= n <= 36:
        return 3
    return None

class ModeloIAHistGB:
    def __init__(self, tipo="duzia", janela=20):
        self.tipo = tipo
        self.janela = janela
        self.modelo = None
        self.encoder = LabelEncoder()
        self.treinado = False

    def construir_features(self, numeros):
        ultimos = numeros[-self.janela:]
        atual = ultimos[-1]
        anteriores = ultimos[:-1]
        grupo = get_duzia(atual)

        features = [
            atual % 2,
            int(str(atual)[-1]),
            atual % 3,
            abs(atual - anteriores[-1]) if anteriores else 0,
            int(atual == anteriores[-1]) if anteriores else 0,
            1 if atual > anteriores[-1] else -1 if atual < anteriores[-1] else 0,
            sum(1 for x in anteriores[-3:] if grupo == get_duzia(x)),
            Counter(numeros[-30:]).get(atual, 0),
            int(atual in [n for n, _ in Counter(numeros[-30:]).most_common(5)]),
            int(np.mean(anteriores) < atual),
            int(atual == 0),
            grupo,
        ]
        freq = Counter(get_duzia(n) for n in numeros[-20:])
        features.append(freq.get(grupo, 0))
        return features

    def treinar(self, historico):
        numeros = [h["number"] for h in historico if 0 <= h["number"] <= 36]
        X, y = [], []
        for i in range(self.janela, len(numeros) - 1):
            janela = numeros[i - self.janela:i + 1]
            target = get_duzia(numeros[i])
            if target is not None:
                X.append(self.construir_features(janela))
                y.append(target)
        if X:
            X = np.array(X, dtype=np.float32)
            y = self.encoder.fit_transform(y)
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.modelo = HistGradientBoostingClassifier(max_iter=200, max_depth=5, random_state=42)
            self.modelo.fit(X, y)
            self.treinado = True

    def prever(self, historico):
        if not self.treinado:
            return None
        numeros = [h["number"] for h in historico if 0 <= h["number"] <= 36]
        if len(numeros) < self.janela + 1:
            return None
        janela = numeros[-(self.janela + 1):]
        entrada = np.array([self.construir_features(janela)], dtype=np.float32)
        proba = self.modelo.predict_proba(entrada)[0]
        if max(proba) >= 0.4:
            return self.encoder.inverse_transform([np.argmax(proba)])[0]
        return None

modelo_global = ModeloIAHistGB()
historico_global = []

@app.on_event("startup")
def carregar_e_treinar():
    global modelo_global, historico_global
    print("[INIT] Carregando histórico e treinando modelo...")

    if not os.path.exists(HISTORICO_PATH):
        print(f"[ERRO] Arquivo não encontrado: {HISTORICO_PATH}")
        return

    with open(HISTORICO_PATH, "r") as f:
        historico_global = json.load(f)

    if len(historico_global) < 5:
        print("[ERRO] Histórico insuficiente para treinar.")
        return

    modelo_global = ModeloIAHistGB()
    modelo_global.treinar(historico_global)

    if modelo_global.treinado:
        print("[OK] Modelo treinado com sucesso.")
    else:
        print("[ERRO] Falha ao treinar modelo.")

@app.get("/previsao-duzia")
def previsao_duzia():
    try:
        if not modelo_global.treinado:
            raise HTTPException(status_code=503, detail="Modelo não está treinado.")
        previsao = modelo_global.prever(historico_global)
        return {"duzia_prevista": to_python(previsao)}
    except HTTPException as http_err:
        print(f"[HTTP ERROR] {http_err.detail}")
        raise http_err
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.post("/atualizar-modelo")
def atualizar_modelo():
    try:
        if not os.path.exists(HISTORICO_PATH):
            raise HTTPException(status_code=404, detail="Arquivo de histórico não encontrado.")
        with open(HISTORICO_PATH, "r") as f:
            historico = json.load(f)
        if len(historico) < 5:
            raise HTTPException(status_code=422, detail="Histórico insuficiente.")
        modelo_global.treinar(historico)
        if modelo_global.treinado:
            global historico_global
            historico_global = historico
            return {"status": "Modelo atualizado com sucesso."}
        else:
            raise HTTPException(status_code=500, detail="Falha ao treinar o modelo.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar modelo: {str(e)}")
