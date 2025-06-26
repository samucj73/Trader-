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
FIREBASE_COLLECTION = "resultados_duzia"
firebase_db = None
try:
    if not firebase_admin._apps:
        firebase_cred_json = os.getenv("FIREBASE_CREDENTIAL_JSON")
        if firebase_cred_json:
            firebase_dict = json.loads(firebase_cred_json)
            cred = credentials.Certificate(firebase_dict)
            print("[FIREBASE] Inicializado com variável de ambiente.")
        else:
            FIREBASE_CRED_PATH = "firebase-adminsdk-fbsvc-2c717cb6fe.json"
            cred = credentials.Certificate(FIREBASE_CRED_PATH)
            print("[FIREBASE] Inicializado com arquivo local.")
        firebase_admin.initialize_app(cred)
        firebase_db = firestore.client()
        print("[FIREBASE] Conectado ao Firebase com sucesso.")
except Exception as e:
    print(f"[ERRO] Falha ao conectar ao Firebase: {e}")

# === FastAPI ===
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"status": "API de previsão de dúzia ativa!"}

# === Push Notification ===
@app.post("/api/salvar-inscricao")
async def salvar_inscricao(request: Request):
    if not firebase_db:
        raise HTTPException(status_code=500, detail="Firebase não inicializado.")

    body = await request.json()

    # Salvar a inscrição no Firestore, evitando duplicatas
    try:
        docs = firebase_db.collection("subscriptions").where("endpoint", "==", body.get("endpoint")).stream()
        if not any(True for _ in docs):  # Se não existe, insere
            firebase_db.collection("subscriptions").add(body)
            print("[PUSH] Nova inscrição salva no Firebase.")
        else:
            print("[PUSH] Inscrição já existe no Firebase.")
        return {"status": "ok"}
    except Exception as e:
        print("[ERRO] Falha ao salvar inscrição:", e)
        raise HTTPException(status_code=500, detail="Erro ao salvar inscrição no Firebase.")

def enviar_push_para_todos(mensagem):
    if not firebase_db:
        print("[ERRO] Firebase não está disponível.")
        return

    try:
        docs = firebase_db.collection("subscriptions").stream()
        total = 0
        for doc in docs:
            sub = doc.to_dict()
            try:
                webpush(
                    subscription_info=sub,
                    data=json.dumps({
                        "title": "🔮 Nova previsão de Dúzia!",
                        "body": mensagem
                    }),
                    vapid_private_key=VAPID_PRIVATE_KEY,
                    vapid_claims={"sub": "mailto:samu.rcj@gmail.com"}
                )
                print(f"[PUSH] Notificação enviada para {sub.get('endpoint')[:50]}...")
                total += 1
            except WebPushException as e:
                print(f"[ERRO] Falha ao enviar push: {e}")
        print(f"[PUSH] Total de notificações enviadas: {total}")
    except Exception as e:
        print("[ERRO] Falha ao buscar inscrições:", e)

@app.get("/api/enviar-teste")
def enviar_teste():
    enviar_push_para_todos("🧪 Esta é uma notificação de teste.")
    return {"status": "ok"}

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
        numeros = [h["number"] for h in historico if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
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
            print("[IA] Modelo ainda não treinado.")
            return None
        numeros = [h["number"] for h in historico if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
        print(f"[IA] Total de números válidos para previsão: {len(numeros)}")
        if len(numeros) < self.janela + 1:
            print("[IA] Números válidos insuficientes para previsão.")
            return None
        janela = numeros[-(self.janela + 1):]
        entrada = np.array([self.construir_features(janela)], dtype=np.float32)
        proba = self.modelo.predict_proba(entrada)[0]
        print("📊 Probabilidades da IA:", proba)
        if max(proba) >= 0.25:
            return self.encoder.inverse_transform([np.argmax(proba)])[0]
        return None

modelo_global = ModeloIAHistGB()
historico_global = []

def carregar_historico():
    try:
        registros = []
        if firebase_db:
            docs = firebase_db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
            for doc in docs:
                dado = doc.to_dict()
                if (
                    isinstance(dado.get("number"), int)
                    and 0 <= dado["number"] <= 36
                    and "timestamp" in dado
                ):
                    registros.append(dado)
                else:
                    print(f"[IGNORADO] Registro inválido no Firebase: {dado}")
        elif os.path.exists(HISTORICO_PATH):
            with open(HISTORICO_PATH) as f:
                dados = json.load(f)
                for dado in dados:
                    if (
                        isinstance(dado.get("number"), int)
                        and 0 <= dado["number"] <= 36
                        and "timestamp" in dado
                    ):
                        registros.append(dado)
                    else:
                        print(f"[IGNORADO] Registro inválido no arquivo local: {dado}")

        # Remover duplicatas por timestamp
        visto = set()
        historico_filtrado = []
        for r in registros:
            if r["timestamp"] not in visto:
                visto.add(r["timestamp"])
                historico_filtrado.append(r)
            else:
                print(f"[DUPLICATA] Ignorando registro com timestamp duplicado: {r['timestamp']}")

        print(f"[HISTÓRICO] Registros válidos carregados: {len(historico_filtrado)}")
        return historico_filtrado

    except Exception as e:
        print(f"[ERRO] Falha ao carregar histórico: {e}")
        return []

def salvar_no_firebase(resultado):
    try:
        if firebase_db:
            doc_id = resultado["timestamp"]
            doc_ref = firebase_db.collection(FIREBASE_COLLECTION).document(doc_id)
            if not doc_ref.get().exists:
                doc_ref.set(resultado)
                print("[FIREBASE] Resultado salvo no Firebase.")
        else:
            dados = []
            if os.path.exists(HISTORICO_PATH):
                with open(HISTORICO_PATH, "r") as f:
                    dados = json.load(f)
            if resultado["timestamp"] not in [d.get("timestamp") for d in dados]:
                dados.append(resultado)
                with open(HISTORICO_PATH, "w") as f2:
                    json.dump(dados, f2, indent=2)
    except Exception as e:
        print(f"[ERRO] Falha ao salvar resultado: {e}")

@app.on_event("startup")
def startup():
    global historico_global, modelo_global
    historico_global = carregar_historico()
    validos = [h for h in historico_global if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
    if len(validos) >= 25:
        try:
            if os.path.exists(MODELO_PATH):
                modelo_global = joblib.load(MODELO_PATH)
                modelo_global.treinado = True
                print("[IA] Modelo carregado de disco.")
            else:
                modelo_global.treinar(validos)
                print("[IA] Modelo treinado novo.")
        except Exception as e:
            print(f"[ERRO] Falha ao carregar modelo: {e}")
    else:
        print(f"[ERRO] Histórico insuficiente. Apenas {len(validos)} registros válidos.")

ultima_previsao = None

@app.get("/previsao-duzia")
def previsao_duzia():
    global historico_global, ultima_previsao
    novo_historico = carregar_historico()
    validos = [h for h in novo_historico if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
    if len(validos) < 25:
        raise HTTPException(status_code=422, detail=f"Histórico insuficiente. Apenas {len(validos)} números válidos.")
    if len(novo_historico) != len(historico_global):
        print("[IA] Histórico atualizado, re-treinando...")
        modelo_global.treinar(novo_historico)
        historico_global = novo_historico
    nova = modelo_global.prever(historico_global)
    if nova is None:
        raise HTTPException(status_code=422, detail="Falha ao gerar previsão.")
    if nova != ultima_previsao:
        enviar_push_para_todos(f"Dúzia prevista: {nova}")
        ultima_previsao = nova
    return {"duzia_prevista": to_python(nova)}

@app.get("/ver-historico")
def ver_historico():
    try:
        dados = carregar_historico()
        return {"total": len(dados), "historico": dados}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler histórico: {str(e)}")

from captura_api import fetch_latest_result

async def loop_captura_automatica():
    global historico_global, ultima_previsao, modelo_global
    while True:
        print("[AUTO] Capturando resultado automaticamente...")
        resultado = fetch_latest_result()
        if resultado:
            if resultado["number"] is not None and 0 <= resultado["number"] <= 36:
                salvar_no_firebase(resultado)
                
                novo_historico = carregar_historico()
                validos = [h for h in novo_historico if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
                
                if len(validos) >= 25:
                    # Só atualiza e treina se histórico mudou
                    if len(novo_historico) != len(historico_global):
                        print("[IA] Histórico atualizado, re-treinando modelo...")
                        modelo_global.treinar(novo_historico)
                        historico_global = novo_historico
                        
                        nova = modelo_global.prever(historico_global)
                        if nova is not None and nova != ultima_previsao:
                            enviar_push_para_todos(f"Dúzia prevista: {nova}")
                            ultima_previsao = nova
                else:
                    print(f"[ERRO] Histórico insuficiente para treinar: {len(validos)} números válidos.")
        await asyncio.sleep(60)

@app.on_event("startup")
async def iniciar_loop():
    asyncio.create_task(loop_captura_automatica())



if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Subindo servidor na porta {port}...")
    uvicorn.run("backend_previsao_duzia:app", host="0.0.0.0", port=port, reload=False)
