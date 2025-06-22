from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

from firebase_integration import salvar_resultado_firebase, carregar_historico_firebase
from captura_api import router as captura_router, fetch_latest_result

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(captura_router)

HISTORICO_PATH = "historico_coluna_duzia.json"
MODELO_PATH = "modelo_duzia.joblib"

def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def get_duzia(n):
    if n == 0: return 0
    elif 1 <= n <= 12: return 1
    elif 13 <= n <= 24: return 2
    elif 25 <= n <= 36: return 3
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
            Counter(get_duzia(n) for n in numeros[-20:]).get(grupo, 0)
        ]
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
            self.modelo = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=42)
            self.modelo.fit(X, y)
            self.treinado = True
            joblib.dump(self, MODELO_PATH)

    def prever(self, historico):
        if not self.treinado:
            return None
        numeros = [h["number"] for h in historico if 0 <= h["number"] <= 36]
        if len(numeros) < self.janela + 1:
            return None
        entrada = np.array([self.construir_features(numeros[-(self.janela + 1):])], dtype=np.float32)
        proba = self.modelo.predict_proba(entrada)[0]
        print("📊 Probabilidades da IA:", proba)
        if max(proba) >= 0.25:
            return self.encoder.inverse_transform([np.argmax(proba)])[0]
        return None

modelo_global = ModeloIAHistGB()
historico_global = []

@app.on_event("startup")
def carregar_e_treinar():
    global modelo_global, historico_global
    print("[INIT] Carregando histórico do Firebase...")
    try:
        historico_global = carregar_historico_firebase()
    except Exception as e:
        print(f"[ERRO] Firebase: {e}")
        historico_global = []

    if os.path.exists(MODELO_PATH):
        try:
            modelo_carregado = joblib.load(MODELO_PATH)
            if isinstance(modelo_carregado, ModeloIAHistGB):
                modelo_carregado.treinado = True
                modelo_global = modelo_carregado
                print("[OK] Modelo carregado com sucesso.")
                return
        except Exception as e:
            print(f"[ERRO] Carregar modelo: {e}")

    if len(historico_global) >= 25:
        modelo_global = ModeloIAHistGB()
        modelo_global.treinar(historico_global)
        print("[INFO] Modelo treinado.")
    else:
        print("[ERRO] Histórico insuficiente.")

@app.get("/previsao-duzia")
def previsao_duzia():
    if len(historico_global) < 25:
        raise HTTPException(status_code=422, detail="Histórico insuficiente.")
    previsao = modelo_global.prever(historico_global)
    return {"duzia_prevista": to_python(previsao)}

@app.get("/ver-historico")
def ver_historico():
    return {"total": len(historico_global), "historico": historico_global}

import asyncio

async def loop_captura_automatica():
    while True:
        print("[AUTO] Capturando resultado automaticamente...")
        resultado = fetch_latest_result()
        if resultado:
            salvar_resultado_firebase(resultado)
            historico_global.append(resultado)
            modelo_global.treinar(historico_global)
            if modelo_global.treinado:
                joblib.dump(modelo_global, MODELO_PATH)
        await asyncio.sleep(60)

@app.on_event("startup")
async def iniciar_loop_background():
    asyncio.create_task(loop_captura_automatica())
