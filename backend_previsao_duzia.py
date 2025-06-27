from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import json, os, numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import asyncio
import firebase_admin
from firebase_admin import credentials, firestore
from pywebpush import webpush, WebPushException

# === Firebase ===
FIREBASE_COLLECTION = "resultados"

if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# === FastAPI app ===
app = FastAPI()

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajuste para produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Dados e modelo ===
historico = []
X, y = [], []
modelo = None
le = LabelEncoder()

# === Notificações ===
VAPID_CLAIMS = {
    "sub": "mailto:exemplo@email.com"
}
VAPID_PRIVATE_KEY = os.environ.get("VAPID_PRIVATE_KEY", "seu_private_key_aqui")
VAPID_PUBLIC_KEY = os.environ.get("VAPID_PUBLIC_KEY", "sua_public_key_aqui")
assinantes = []

# === Funções ===
def extrair_features(historico):
    features = []
    for i in range(len(historico) - 1):
        entrada = historico[i]
        anterior = historico[i - 1] if i > 0 else {"numero": 0}
        diff = entrada["numero"] - anterior["numero"]
        features.append([
            entrada["numero"],
            entrada["numero"] % 2,
            entrada["numero"] % 3,
            entrada["numero"] % 12,
            diff
        ])
    return features

def preparar_dados():
    global X, y
    if len(historico) > 1:
        X = extrair_features(historico[:-1])
        y_raw = [get_duzia(h["numero"]) for h in historico[1:]]
        y = le.fit_transform(y_raw)

def treinar_modelo():
    global modelo
    if len(X) > 10:
        modelo = HistGradientBoostingClassifier()
        modelo.fit(X, y)

def get_duzia(numero):
    if 1 <= numero <= 12:
        return "1ª"
    elif 13 <= numero <= 24:
        return "2ª"
    elif 25 <= numero <= 36:
        return "3ª"
    return "Zero"

async def carregar_historico():
    global historico
    docs = db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
    historico = [doc.to_dict() for doc in docs]
    preparar_dados()
    treinar_modelo()

def salvar_no_firestore(dado):
    db.collection(FIREBASE_COLLECTION).add(dado)

async def notificar(duzia):
    payload = json.dumps({"duzia": duzia})
    for sub in assinantes:
        try:
            webpush(
                subscription_info=sub,
                data=payload,
                vapid_private_key=VAPID_PRIVATE_KEY,
                vapid_claims=VAPID_CLAIMS
            )
        except WebPushException as e:
            print("Erro ao enviar notificação:", e)

# === Rotas ===
@app.on_event("startup")
async def startup_event():
    await carregar_historico()

@app.post("/novo_resultado/")
async def novo_resultado(request: Request):
    data = await request.json()
    numero = data.get("numero")

    if numero is None or not (0 <= numero <= 36):
        raise HTTPException(status_code=400, detail="Número inválido")

    entrada = {"numero": numero, "timestamp": firestore.SERVER_TIMESTAMP}
    historico.append({"numero": numero})
    salvar_no_firestore(entrada)

    preparar_dados()
    treinar_modelo()

    return {"mensagem": "Resultado salvo com sucesso"}

@app.get("/prever/")
def prever():
    if modelo is None or len(historico) < 2:
        raise HTTPException(status_code=400, detail="Dados insuficientes")

    ultima = historico[-1]
    anterior = historico[-2] if len(historico) >= 2 else {"numero": 0}
    features = np.array([[
        ultima["numero"],
        ultima["numero"] % 2,
        ultima["numero"] % 3,
        ultima["numero"] % 12,
        ultima["numero"] - anterior["numero"]
    ]])

    pred = modelo.predict(features)[0]
    duzia = le.inverse_transform([pred])[0]

    asyncio.create_task(notificar(duzia))

    return {"previsao": duzia}

@app.post("/subscribe/")
async def subscribe(request: Request):
    sub = await request.json()
    assinantes.append(sub)
    return {"mensagem": "Inscrição recebida com sucesso"}

@app.get("/")
def home():
    return {"mensagem": "API de Previsão de Dúzia funcionando!"}

# === Execução local ou no Render ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend_previsao_duzia:app", host="0.0.0.0", port=port)
