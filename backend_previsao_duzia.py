from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import asyncio
import firebase_admin
from firebase_admin import credentials, firestore

# === Firebase ===
FIREBASE_CRED_PATH = "firebase-adminsdk-fbsvc-2c717cb6fe.json"
FIREBASE_COLLECTION = "resultados_duzia"

firebase_db = None
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        firebase_db = firestore.client()
        print("[FIREBASE] Conectado ao Firebase com sucesso.")
except Exception as e:
    print(f"[ERRO] Falha ao conectar ao Firebase: {e}")

# === FastAPI ===
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# === Arquivos locais ===
HISTORICO_PATH = "historico_coluna_duzia.json"
MODELO_PATH = "modelo_duzia.joblib"

# === IA ===
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
            self.modelo = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=42)
            self.modelo.fit(X, y)
            self.treinado = True
            joblib.dump(self, MODELO_PATH)

    def prever(self, historico):
        if not self.treinado: return None
        numeros = [h["number"] for h in historico if 0 <= h["number"] <= 36]
        if len(numeros) < self.janela + 1: return None
        janela = numeros[-(self.janela + 1):]
        entrada = np.array([self.construir_features(janela)], dtype=np.float32)
        proba = self.modelo.predict_proba(entrada)[0]
        print("📊 Probabilidades da IA:", proba)
        if max(proba) >= 0.25:
            return self.encoder.inverse_transform([np.argmax(proba)])[0]
        return None

modelo_global = ModeloIAHistGB()
historico_global = []

# === Carregar histórico ===
def carregar_historico():
    try:
        if firebase_db:
            docs = firebase_db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
            return [doc.to_dict() for doc in docs]
        elif os.path.exists(HISTORICO_PATH):
            with open(HISTORICO_PATH) as f:
                return json.load(f)
        else:
            return []
    except Exception as e:
        print(f"[ERRO] Falha ao carregar histórico: {e}")
        return []

# === Salvar resultado ===
def salvar_no_firebase(resultado):
    try:
        if firebase_db:
            firebase_db.collection(FIREBASE_COLLECTION).document(resultado["timestamp"]).set(resultado)
            print("[FIREBASE] Resultado salvo no Firebase.")
        else:
            with open(HISTORICO_PATH, "r") as f:
                dados = json.load(f)
            if resultado["timestamp"] not in [d.get("timestamp") for d in dados]:
                dados.append(resultado)
                with open(HISTORICO_PATH, "w") as f2:
                    json.dump(dados, f2, indent=2)
    except Exception as e:
        print(f"[ERRO] Falha ao salvar no Firebase: {e}")

# === Startup: carregar e treinar ===
@app.on_event("startup")
def startup():
    global historico_global, modelo_global
    historico_global = carregar_historico()
    if len(historico_global) >= 25:
        try:
            if os.path.exists(MODELO_PATH):
                modelo_global = joblib.load(MODELO_PATH)
                modelo_global.treinado = True
                print("[IA] Modelo carregado de disco.")
            else:
                modelo_global.treinar(historico_global)
                print("[IA] Modelo treinado novo.")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo: {e}")
    else:
        print("[ERRO] Histórico insuficiente.")

# === Previsão da Dúzia ===
@app.get("/previsao-duzia")
def previsao_duzia():
    global historico_global
    novo_historico = carregar_historico()
    if len(novo_historico) < 25:
        raise HTTPException(status_code=422, detail="Histórico insuficiente.")
    if len(novo_historico) != len(historico_global):
        print("[IA] Histórico atualizado, re-treinando...")
        modelo_global.treinar(novo_historico)
        historico_global = novo_historico
    previsao = modelo_global.prever(historico_global)
    return {"duzia_prevista": to_python(previsao)}

# === Ver histórico ===
@app.get("/ver-historico")
def ver_historico():
    try:
        dados = carregar_historico()
        return {"total": len(dados), "historico": dados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler histórico: {str(e)}")

# === Captura automática ===
from captura_api import fetch_latest_result

async def loop_captura_automatica():
    while True:
        print("[AUTO] Capturando resultado automaticamente...")
        resultado = fetch_latest_result()
        if resultado:
            salvar_no_firebase(resultado)
        await asyncio.sleep(60)

@app.on_event("startup")
async def iniciar_loop():
    asyncio.create_task(loop_captura_automatica())
