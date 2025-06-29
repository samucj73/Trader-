import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Previsão de Dúzia - Roleta IA", layout="centered")

# === Firebase via secrets ===
@st.cache_resource
def init_firebase():
    firebase_key = st.secrets["firebase_key_json"]
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key)
        firebase_admin.initialize_app(cred)
    return firestore.client()

# === Carregar modelo IA ===
@st.cache_resource
def carregar_modelo():
    caminho_modelo = "modelo_duzia.pkl"
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError("Arquivo modelo_duzia.pkl não encontrado!")
    return joblib.load(caminho_modelo)

# === App principal ===
st.title("🎰 Previsão de Dúzia - Roleta IA")

try:
    st.write("🔌 Conectando ao Firebase...")
    db = init_firebase()

    st.write("🧠 Carregando modelo IA...")
    modelo = carregar_modelo()

    st.write("📊 Buscando últimos resultados...")
    colecao = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20)
    docs = colecao.stream()

    historico = []
    for doc in docs:
        data = doc.to_dict()
        if "number" in data:
            historico.append(data["number"])

    if len(historico) < 10:
        st.warning("⚠️ Ainda não há dados suficientes para prever. Adicione pelo menos 10 resultados.")
    else:
        entrada = pd.DataFrame([historico[:10]], columns=[f"n{i}" for i in range(10)])
        predicao = modelo.predict(entrada)[0]
        st.success(f"🔮 Previsão da próxima dúzia: **{predicao}**")

        # Mostrar histórico
        st.markdown("### 📌 Últimos números:")
        st.write(historico)

except Exception as e:
    st.error(f"❌ Erro ao rodar o aplicativo:\n\n`{e}`")
