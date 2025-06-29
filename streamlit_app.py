import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore
import json
import joblib
import pandas as pd
import os
from sklearn.ensemble import HistGradientBoostingClassifier

st.set_page_config(page_title="Previsão de Dúzia", layout="centered")

# === Firebase - usando chave embutida ===
@st.cache_resource
def init_firebase():
    firebase_key_json = {
        "type": "service_account",
        "project_id": "roleta-ia-duzia",
        "private_key_id": "2c717cb6feb5a9678d1d9a23d5349f55c7a6bb32",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCYOcX/+QLEzwNP\n...[TRUNCADO]...\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk-fbsvc@roleta-ia-duzia.iam.gserviceaccount.com",
        "client_id": "100348573633888104841",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40roleta-ia-duzia.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }

    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_json)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# === Carregar modelo ===
@st.cache_resource
def carregar_modelo():
    caminho_modelo = "modelo_duzia.pkl"
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError("Arquivo modelo_duzia.pkl não encontrado!")
    return joblib.load(caminho_modelo)

# === App principal ===
st.title("Previsão de Dúzia - Roleta IA")

try:
    st.write("🔌 Conectando ao Firebase...")
    db = init_firebase()

    st.write("🧠 Carregando modelo IA...")
    modelo = carregar_modelo()

    st.write("📊 Carregando dados do Firestore...")
    colecao = db.collection("resultados").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20)
    docs = colecao.stream()

    historico = []
    for doc in docs:
        data = doc.to_dict()
        if "number" in data:
            historico.append(data["number"])

    if len(historico) < 10:
        st.warning("⚠️ Ainda não há dados suficientes para prever. Adicione mais resultados.")
    else:
        entrada = pd.DataFrame([historico[:10]], columns=[f"n{i}" for i in range(10)])
        predicao = modelo.predict(entrada)[0]
        st.success(f"🔮 Previsão da próxima dúzia: **{predicao}**")

        # Exibir últimos números
        st.markdown("### 🎯 Últimos números:")
        st.write(historico)

except Exception as e:
    st.error(f"❌ Erro ao rodar o aplicativo:\n\n`{e}`")
