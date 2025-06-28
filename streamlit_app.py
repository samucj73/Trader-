import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import numpy as np
import joblib
import json
from streamlit.runtime.secrets import secrets

# Inicializa Firebase via secrets
@st.cache_resource
def init_firebase():
    cred_dict = json.loads(secrets["firebase_key"])
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    return firestore.client()

# Carrega modelo de IA
@st.cache_resource
def load_model():
    return joblib.load("modelo_duzia.pkl")

# Classifica número na dúzia
def classificar_duzia(n):
    if n == 0: return 0
    if 1 <= n <= 12: return 1
    if 13 <= n <= 24: return 2
    return 3

# Previsão com os últimos 5 números
def prever_duzia(modelo, ultimos):
    x = np.array(ultimos[-5:]).reshape(1, -1)
    return modelo.predict(x)[0]

# App Streamlit
st.set_page_config(page_title="Previsão de Dúzia", layout="centered")
st.title("🎰 Previsão da Dúzia na Roleta")

# Inicia Firebase e carrega modelo
db = init_firebase()
model = load_model()

# Busca dados do Firestore
docs = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
resultados = [d.to_dict() for d in docs]
resultados.sort(key=lambda r: r["timestamp"])

# Lista de números sorteados
numeros = [r["number"] for r in resultados]
st.write("🎯 Últimos números:", numeros)

# Previsão e acerto
if len(numeros) >= 5:
    previsao = prever_duzia(model, numeros)
    st.subheader(f"🔮 Próxima dúzia prevista: **{previsao}**")

    real = classificar_duzia(numeros[-1])
    if previsao == real:
        st.success(f"✅ Acertou! O número {numeros[-1]} pertence à dúzia {real}.")
    else:
        st.error(f"❌ Errou! O número {numeros[-1]} pertence à dúzia {real}.")

    # Contagem de acertos simulados
    acertos = sum(
        prever_duzia(model, numeros[i-5:i]) == classificar_duzia(numeros[i])
        for i in range(5, len(numeros))
    )
    st.info(f"📊 Acertos simulados: {acertos} de {len(numeros)-5}")
else:
    st.warning("🔄 Aguardando mais resultados para prever.")
