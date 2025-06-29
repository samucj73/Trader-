import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import joblib
import pandas as pd
import os

st.set_page_config(page_title="Previsão de Dúzia - Roleta IA", layout="centered")

# === Inicializar Firebase corretamente ===
@st.cache_resource
def init_firebase():
    # Corrige o erro forçando a conversão para dict
    firebase_key_dict = dict(st.secrets["firebase_key_json"])

    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_dict)
        firebase_admin.initialize_app(cred)
    return firestore.client()

# === Carrega modelo IA ===
@st.cache_resource
def carregar_modelo():
    caminho = "modelo_duzia.pkl"
    if not os.path.exists(caminho):
        raise FileNotFoundError("❌ Arquivo 'modelo_duzia.pkl' não encontrado.")
    return joblib.load(caminho)

# === App principal ===
st.title("🎰 Previsão de Dúzia - Roleta IA")

try:
    db = init_firebase()
    modelo = carregar_modelo()

    st.write("📊 Buscando últimos resultados...")
    colecao = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20)
    docs = colecao.stream()

    historico = [doc.to_dict()["number"] for doc in docs if "number" in doc.to_dict()]

    if len(historico) < 10:
        st.warning("⚠️ Ainda não há dados suficientes para prever. Adicione pelo menos 10 resultados.")
    else:
        entrada = pd.DataFrame([historico[:10]], columns=[f"n{i}" for i in range(10)])
        predicao = modelo.predict(entrada)[0]
        st.success(f"🔮 Previsão da próxima dúzia: **{predicao}**")

        st.markdown("### 📌 Últimos números:")
        st.write(historico)

except Exception as e:
    st.error(f"❌ Erro ao rodar o app:\n\n```\n{e}\n```")
