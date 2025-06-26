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
from captura_api import fetch_latest_result  # Certifique-se de que este arquivo está presente
from contextlib import asynccontextmanager  # <== ADICIONADO para lifespan

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

# === Lifespan ===
@asynccontextmanager
async def lifespan(app: FastAPI):
    global historico_global, modelo_global
    historico_global = carregar_historico()
    validos = [h for h in historico_global if isinstance(h["number"], int)]
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
    asyncio.create_task(loop_captura_automatica())
    yield

# === FastAPI ===
app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def home():
    return {"status": "API de previsão de dúzia ativa!"}

# === Push Notification ===
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "SUA_CHAVE_PRIVADA_AQUI")

@app.post("/api/salvar-inscricao")
async def salvar_inscricao(request: Request):
    if not firebase_db:
        raise HTTPException(status_code=500, detail="Firebase não inicializado.")
    body = await request.json()
    try:
        docs = firebase_db.collection("subscriptions").where("endpoint", "==", body.get("endpoint")).stream()
        if not any(True for _ in docs):
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
    def __init__(self, tipo="duzia", janela=250):
        self.tipo = tipo
        self.janela = janela
        self.modelo = None
        self.encoder = LabelEncoder()
        self.treinado = False

    def construir_features(self, numeros):
        ultimos = numeros[-self.janela:]
        atual = ultimos[-1]
        anteriores = ultimos[:-1]

        def safe_get_duzia(n):
            return -1 if n == 0 else get_duzia(n)

        grupo = safe_get_duzia(atual)

        freq_20 = Counter(safe_get_duzia(n) for n in numeros[-20:])
        freq_50 = Counter(safe_get_duzia(n) for n in numeros[-50:]) if len(numeros) >= 50 else freq_20
        total_50 = sum(freq_50.values()) or 1

        lag1 = safe_get_duzia(anteriores[-1]) if len(anteriores) >= 1 else -1
        lag2 = safe_get_duzia(anteriores[-2]) if len(anteriores) >= 2 else -1
        lag3 = safe_get_duzia(anteriores[-3]) if len(anteriores) >= 3 else -1

        val1 = anteriores[-1] if len(anteriores) >= 1 else 0
        val2 = anteriores[-2] if len(anteriores) >= 2 else 0
        val3 = anteriores[-3] if len(anteriores) >= 3 else 0

        tendencia = 0
        if len(anteriores) >= 3:
            diffs = np.diff(anteriores[-3:])
            tendencia = int(np.mean(diffs) > 0) - int(np.mean(diffs) < 0)

        zeros_50 = numeros[-50:].count(0)
        porc_zeros = zeros_50 / 50

        densidade_20 = freq_20.get(grupo, 0)
        densidade_50 = freq_50.get(grupo, 0)
        rel_freq_grupo = densidade_50 / total_50

        repete_duzia = int(grupo == safe_get_duzia(anteriores[-1])) if anteriores else 0

        features = [
            atual % 2,
            atual % 3,
            int(str(atual)[-1]),
            abs(atual - anteriores[-1]) if anteriores else 0,
            int(atual == anteriores[-1]) if anteriores else 0,
            1 if atual > anteriores[-1] else -1 if atual < anteriores[-1] else 0,
            sum(1 for x in anteriores[-3:] if grupo == safe_get_duzia(x)),
            Counter(numeros[-30:]).get(atual, 0),
            int(atual in [n for n, _ in Counter(numeros[-30:]).most_common(5)]),
            int(np.mean(anteriores) < atual),
            int(atual == 0),
            grupo,
            densidade_20,
            densidade_50,
            rel_freq_grupo,
            repete_duzia,
            tendencia,
            lag1, lag2, lag3,
            val1, val2, val3,
            porc_zeros
        ]

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
        if not X:
            print("[IA] Dados insuficientes para treino.")
            return

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        y_enc = self.encoder.fit_transform(y)

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_enc[:split_idx], y_enc[split_idx:]

        self.modelo = HistGradientBoostingClassifier(max_iter=200, max_depth=7, random_state=42)
        self.modelo.fit(X_train, y_train)
        self.treinado = True

        acc = self.modelo.score(X_test, y_test)
        print(f"[IA] Modelo treinado. Acurácia no teste: {acc:.3f}")

        joblib.dump(self, MODELO_PATH)

    def prever(self, historico):
        if not self.treinado:
            print("[IA] Modelo ainda não treinado.")
            return None
        numeros = [h["number"] for h in historico if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
        if len(numeros) < self.janela + 1:
            print("[IA] Números válidos insuficientes para previsão.")
            return None
        janela = numeros[-(self.janela + 1):]
        entrada = np.array([self.construir_features(janela)], dtype=np.float32)
        proba = self.modelo.predict_proba(entrada)[0]
        print(f"📊 Probabilidades da IA: {proba}")
        pred = np.argmax(proba)
        return self.encoder.inverse_transform([pred])[0]

modelo_global = ModeloIAHistGB()
historico_global = []
ultima_previsao = None

def carregar_historico():
    try:
        registros = []
        if firebase_db:
            docs = firebase_db.collection(FIREBASE_COLLECTION).order_by("timestamp").stream()
            for doc in docs:
                dado = doc.to_dict()
                if isinstance(dado.get("number"), int) and 0 <= dado["number"] <= 36 and "timestamp" in dado:
                    registros.append(dado)
        elif os.path.exists(HISTORICO_PATH):
            with open(HISTORICO_PATH) as f:
                dados = json.load(f)
                for dado in dados:
                    if isinstance(dado.get("number"), int) and 0 <= dado["number"] <= 36 and "timestamp" in dado:
                        registros.append(dado)

        visto = set()
        historico_filtrado = []
        for r in registros:
            if r["timestamp"] not in visto:
                visto.add(r["timestamp"])
                historico_filtrado.append(r)
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
    except Exception as e:
        print(f"[ERRO] Falha ao salvar resultado: {e}")

@app.get("/previsao-duzia")
def previsao_duzia():
    global historico_global, ultima_previsao
    novo_historico = carregar_historico()
    validos = [h for h in novo_historico if isinstance(h["number"], int) and 0 <= h["number"] <= 36]
    if len(validos) < 25:
        raise HTTPException(status_code=422, detail="Histórico insuficiente.")
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

# === Loop automático ===
    #async def loop_captura_automatica():
    global historico_global, ultima_previsao, modelo_global
    while True:
        print("[AUTO] Capturando resultado automaticamente...")
        resultado = fetch_latest_result()
        if resultado and 0 <= resultado["number"] <= 36:
            salvar_no_firebase(resultado)
            novo_historico = carregar_historico()
            validos = [h for h in novo_historico if isinstance(h["number"], int)]
            if len(novo_historico) != len(historico_global):
                modelo_global.treinar(novo_historico)
                historico_global = novo_historico
                nova = modelo_global.prever(historico_global)
                if nova and nova != ultima_previsao:
                    enviar_push_para_todos(f"Dúzia prevista: {nova}")
                    ultima_previsao = nova
        await asyncio.sleep(60)

# === Execução local ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Subindo servidor na porta {port}...")
    uvicorn.run("backend_previsao_duzia:app", host="0.0.0.0", port=port, reload=False)
